import logging
import math
import os
import pickle
import time
from glob import glob
from io import StringIO
from pathlib import Path
from shutil import copyfile
from typing import Tuple

import numpy as np
import pandas as pd
import pysam
from bitarray import bitarray
from pandarallel import pandarallel
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from chowderx.CellRangerIO import CellRangerIO
from chowderx.ChowderIO import ChowderIO

logger = logging.getLogger()


class ChowderX:
    S3_BUCKET_PATH = "<S3_BUCKET_PATH>"
    S3_BATCH_DIRS = {}
    BAM_FILE_NAME = "possorted_genome_bam.bam"
    BAI_FILE_NAME = "possorted_genome_bam.bam.bai"
    BAM_DIR_LOCAL_PATH = "/data/bam_files"
    DEMUXLET_RELATIVE_PATH = "demuxlet/demuxlet.best"
    TENX_RELATIVE_PATH = "cellranger_count/outs"
    TENX_FILTERED_SUBDIR = "filtered_feature_bc_matrix"
    TENX_RAW_SUBDIR = "raw_feature_bc_matrix"
    VCF_FILEPATH = "<VCF_FILEPATH>"
    VCF_BASECOLS = ["CHROM", "POS", "ID", "REF", "ALT"]
    MIN_CELLS_PER_DONOR = 50
    PBAR_CHUNK_SIZE = 1000000  # with 1000, there was overflow because max size is (~1M?)
    GENOTYPES = {"0/0", "0/1", "1/1"}
    HOMOZYGOUS = {"0/0", "1/1"}

    def __init__(self, batch, lane):
        self.batch = batch
        self.lane = lane

    def run(self):
        # check for things to not store with `self` to save memory
        run_start_time = time.time()
        log_str = f"batch {self.batch}, lane {self.lane}: "
        self.failed_qc = 0
        self._connect_to_files()

        logger.info(log_str + "loading counts matrices...")
        self.donors = self._get_donors_in_tenx_lane(self.lane)
        self._cell_id_donor_lookup = self._get_cell_id_donor_lookup(self.demuxlet_path)
        self.features, drop_ids_df, self.counts_raw = CellRangerIO.load_dir(
            self.tenx_raw_path, sparse=True, merged=False
        )
        self.drop_ids = set(drop_ids_df)
        self.cell_ids = set(self._cell_id_donor_lookup.keys())
        self.soup_ids, self.soup_expr, self.soup_expr_prof = self._get_soup_ids_and_expr(drop_ids_df)

        logger.info(log_str + "assimilating donor variants...")
        self._vcf = self._load_vcf(self.donors, self.VCF_FILEPATH)
        self._global_ref_alleles = self._build_global_ref_alleles(self._vcf)
        self._donor_var_vcf_dict = self._build_donor_subset_vcf_dict(self._vcf, self.donors, "variable")
        self._donor_unq_vcf_dict = self._build_donor_subset_vcf_dict(self._vcf, self.donors, "unique")
        self._global_var_vcf = self._get_global_subset_vcf(self._donor_var_vcf_dict)
        self._global_unq_vcf = self._get_global_subset_vcf(self._donor_unq_vcf_dict)

        logger.info(log_str + "building snp lookup...")
        self._donor_var_snp_bitarr_dict = self._build_donor_snp_bitarray_dict(self._donor_var_vcf_dict)
        self._donor_unq_snp_bitarr_dict = self._build_donor_snp_bitarray_dict(self._donor_unq_vcf_dict)
        self._global_var_snp_bitarr_dict = self._build_snp_bitarray_dict(self._global_var_vcf)
        self._global_unq_snp_bitarr_dict = self._build_snp_bitarray_dict(self._global_unq_vcf)
        self._donor_var_allele_dict = self._build_donor_allele_dict(self._donor_var_vcf_dict)
        self._donor_unq_allele_dict = self._build_donor_allele_dict(self._donor_unq_vcf_dict)
        self._global_var_cum_snp_arr_dict = self._build_global_cum_snp_arr_dict(self._global_var_snp_bitarr_dict)
        self._global_unq_cum_snp_arr_dict = self._build_global_cum_snp_arr_dict(self._global_unq_snp_bitarr_dict)

        logger.info(log_str + "parsing bam file...")
        self.bam_stats_df = self._get_bam_stats(str(self._bam_local_path))
        self._cell_snp_reads, self._soup_snp_reads = self._get_snp_containing_reads(
            self._bam, self._global_var_cum_snp_arr_dict, self.cell_ids, self.soup_ids
        )

        logger.info(log_str + "extracting snp data...")
        (
            self._cell_nativity_lookup,
            self._donor_soup_nativity_lookup,
            self.soup_snp_df,
            self.donor_cell_snp_df,
        ) = self._extract_snp_data(self._cell_snp_reads, self._soup_snp_reads, self._cell_id_donor_lookup)

        logger.info(log_str + "loading 10X data...")
        self.counts = CellRangerIO.load_dir(self.tenx_filtered_path, sparse=False, merged=True)

        logger.info(log_str + "building dataframes...")
        self.cell_df = self._build_cell_df(
            self._cell_id_donor_lookup, self._cell_nativity_lookup, self.cell_ids, self.counts
        )
        self.donor_df = self._build_donor_df(self._donor_soup_nativity_lookup, self.donors)
        self.cell_df, self.donor_df, self.donor_cell_snp_df, self.donors = self.remove_bad_donors(
            self.cell_df, self.donor_df, self.donor_cell_snp_df, self.MIN_CELLS_PER_DONOR
        )
        self.donor_norm_factors = self.estimate_donor_norm_factors(self.soup_snp_df, self.donor_cell_snp_df)

        logger.info(log_str + "estimating and correcting contamination...")
        self.cell_df = self.estimate_contamination(self.cell_df, self.donor_df)
        # self.corrected_expression = self.correct_expression(
        #     self.counts, self.cell_num_contaminated_counts, self.soup_expr_prof
        # )

        self.donor_df.index = pd.MultiIndex.from_product([self.donor_df.index, [self.lane]])
        self.bam_stats_df.index = pd.MultiIndex.from_product([self.bam_stats_df.index, [self.lane]])
        self.runtime = time.time() - run_start_time
        logger.info(f"ChowderX ran in {self.runtime}")

    def _connect_to_files(self):
        # TODO: change all this to a dict, then use kwargs to allow any of it to be overwritten, then use setattr
        self.lane_path = self.get_lane_path(self.S3_BUCKET_PATH, self.batch, self.lane)
        self.demuxlet_path = self.lane_path / self.DEMUXLET_RELATIVE_PATH
        self.tenx_path = self.lane_path / self.TENX_RELATIVE_PATH
        self.tenx_raw_path = self.tenx_path / self.TENX_RAW_SUBDIR
        self.tenx_filtered_path = self.tenx_path / self.TENX_FILTERED_SUBDIR
        self._bam_path_remote = self.tenx_path / self.BAM_FILE_NAME
        self._bai_path_remote = self.tenx_path / self.BAI_FILE_NAME
        self._bam_local_dir = Path(self.BAM_DIR_LOCAL_PATH) / f"{self.batch}/{self.lane}"
        self._bam_local_path = self._bam_local_dir / self.BAM_FILE_NAME
        self._bai_local_path = self._bam_local_dir / self.BAI_FILE_NAME
        if not os.path.exists(self._bam_local_path):
            os.makedirs(self._bam_local_dir, exist_ok=True)
            copyfile(self._bam_path_remote, self._bam_local_path)
        if not os.path.exists(self._bai_local_path):
            os.makedirs(self._bam_local_dir, exist_ok=True)
            copyfile(self._bai_path_remote, self._bai_local_path)
        self._bam = pysam.AlignmentFile(self._bam_local_path, "rb")

    @staticmethod
    def _load_vcf(donors, vcf_filepath):
        with open(vcf_filepath, "r") as f:
            for row in f:
                if row.startswith("#C"):
                    names = row[1:].split("\t")
                    break
        vcf = pd.read_csv(vcf_filepath, names=names, comment="#", delimiter="\t")
        vcf = vcf[ChowderX.VCF_BASECOLS + donors]
        return vcf

    @staticmethod
    def _build_donor_subset_vcf_dict(vcf, donors, method):
        "subset VCF for SNPs for which `donor` is homozygous and distinguished from other `donors`"

        def _row_conditional_variable(row):
            donor_genotype = row[donor]
            other_genotypes = ChowderX.GENOTYPES.difference({donor_genotype})
            if donor_genotype not in ChowderX.HOMOZYGOUS:
                return False
            if not any(row[donors].isin(other_genotypes)):
                return False
            else:
                return True

        def _row_conditional_unique(row):
            donor_genotype = row[donor]
            other_donors = list(set(donors).difference({donor}))
            if donor_genotype in ChowderX.HOMOZYGOUS:
                other_homo_genotype = ChowderX.HOMOZYGOUS.difference({donor_genotype})
                if all(row[other_donors].isin(other_homo_genotype)):
                    return True
            else:
                all_others_match = any((all(row[other_donors].isin({genotype})) for genotype in ChowderX.HOMOZYGOUS))
                if all_others_match:
                    return True
            return False

        if method == "variable":
            row_cond = _row_conditional_variable
        elif method == "unique":
            row_cond = _row_conditional_unique
            vcf = vcf[(vcf == "./.").sum(axis=1) == 0]
        pandarallel.initialize()
        donor_subset_vcf_dict = dict()
        for donor in donors:
            donor_subset_vcf_dict[donor] = vcf[vcf.parallel_apply(lambda row: row_cond(row), axis=1)]
        return donor_subset_vcf_dict

    @staticmethod
    def _get_global_subset_vcf(donor_subset_vcf_dict):
        global_variable_vcf = pd.concat(donor_subset_vcf_dict.values()).drop_duplicates()
        return global_variable_vcf

    def _build_donor_snp_bitarray_dict(self, donor_subset_vcf_dict):
        """ """
        # TODO: Parallelize this (replace dict comp with joblib)
        donor_snp_bitarray_dict = {
            donor: self._build_snp_bitarray_dict(donor_vcf) for donor, donor_vcf in donor_subset_vcf_dict.items()
        }
        return donor_snp_bitarray_dict

    def _build_snp_bitarray_dict(self, vcf):
        # TODO: reimplement as max projection
        def _ints_to_bitarray(snp_inds, arr_size):
            arr = bitarray(arr_size + 1)
            arr.setall(False)
            for i in snp_inds:
                arr[i] = 1
            return arr

        chrom_snp_inds_dict = vcf[["CHROM", "POS"]].groupby("CHROM")["POS"].apply(list).to_dict()
        lengths = self._bam.lengths
        chroms = self._bam.references
        snp_bitarray = {
            chrom: _ints_to_bitarray(chrom_snp_inds_dict[chrom], lengths[i] + 1)
            for i, chrom in enumerate(chroms)
            if chrom in chrom_snp_inds_dict
        }
        return snp_bitarray

    @staticmethod
    def _build_donor_allele_dict(donor_subset_vcf_dict):
        """ """

        def _build_allele_dict(donor, donor_vcf):
            vcf_wt = donor_vcf[donor_vcf[donor] == "0/0"]
            vcf_mut = donor_vcf[donor_vcf[donor] == "1/1"]
            base_dict = vcf_wt[["CHROM", "POS", "REF"]].set_index(["CHROM", "POS"]).to_dict()["REF"]
            base_dict.update(vcf_mut[["CHROM", "POS", "ALT"]].set_index(["CHROM", "POS"]).to_dict()["ALT"])
            return base_dict

        donor_allele_dict = {donor: _build_allele_dict(donor, vcf) for donor, vcf in donor_subset_vcf_dict.items()}
        return donor_allele_dict

    @staticmethod
    def _build_global_ref_alleles(vcf):
        return pd.Series(vcf[["CHROM", "POS", "REF"]].set_index(["CHROM", "POS"]).to_dict()["REF"])

    @staticmethod
    def _get_cell_id_donor_lookup(demuxlet_path):
        demux = pd.read_csv(demuxlet_path, delimiter="\t")
        donor_cells = demux[demux.BEST.apply(lambda x: x.startswith("SNG"))][["BARCODE", "SNG.1ST"]]
        donor_cells.index = donor_cells.BARCODE
        del donor_cells["BARCODE"]
        donor_lookup = donor_cells.to_dict()["SNG.1ST"]
        return donor_lookup

    def _get_soup_ids_and_expr(self, drop_ids_df):
        drop_ids_df = drop_ids_df["barcode"]
        soup_ids_df = drop_ids_df[~drop_ids_df.isin(self.cell_ids)]
        mat_soup = self.counts_raw[soup_ids_df.index]
        soup_expr_arr = np.array(mat_soup.sum(axis=0))[0]
        soup_expr_df = pd.DataFrame(
            {
                "count": soup_expr_arr,
                "ensembl_id": self.features["ensembl_id"],
                "gene_name": self.features["gene_name"],
            }
        )
        soup_expr = soup_expr_df.groupby("gene_name").sum()
        soup_expr_prof = soup_expr / soup_expr.sum()
        soup_ids = set(soup_ids_df)
        return soup_ids, soup_expr, soup_expr_prof

    @staticmethod
    def _build_global_cum_snp_arr_dict(global_snp_bitarray_dict):
        global_cum_snp_arr_dict = {
            donor: np.cumsum(bitarr.tolist()) for donor, bitarr in global_snp_bitarray_dict.items()
        }
        return global_cum_snp_arr_dict

    def _get_snp_containing_reads(self, bam, global_cum_snp_arr_dict, cell_ids, soup_ids):
        """ """

        def _worker(read):
            if not (read.has_tag("CB") and read.has_tag("UB")):
                return None
            drop_id = read.get_tag("CB")
            if (drop_id not in cell_ids) and (drop_id not in soup_ids):
                return None
            elif not (read.get_tag("AS") >= read.rlen - 8 and read.get_tag("NH") == 1):
                return None
            ref = read.reference_name
            if ref not in global_cum_snp_arr_dict:
                return None
            global_cum_snp_arr = global_cum_snp_arr_dict[read.reference_name]
            for (start, end) in read.blocks:
                if global_cum_snp_arr[start] < global_cum_snp_arr[end + 1]:
                    return read

        n_reads = self.bam_stats_df[["mapped", "unmapped"]].sum().sum()
        pbar_chunk = 10000
        pbar_size = math.ceil(n_reads / pbar_chunk)
        count = 0
        bam_filtered = []
        with tqdm(total=pbar_size) as pbar:
            for read_ in bam:
                bam_filtered.append(_worker(read_))
                count += 1
                if count % pbar_chunk == 0:
                    pbar.update(1)
        cell_snp_reads, soup_snp_reads = self._get_idxs_snp_reads(bam_filtered)
        return cell_snp_reads, soup_snp_reads

    def _get_idxs_snp_reads(self, bam_filtered):
        cell_snp_reads = []
        soup_snp_reads = []
        for read in bam_filtered:
            if read is None:
                continue
            chrom = read.reference_name
            drop_id = read.get_tag("CB")
            if drop_id in self.cell_ids:
                is_cell = True
                donor = self._cell_id_donor_lookup[drop_id]
                bitarr = self._donor_var_snp_bitarr_dict[donor][chrom]
            else:
                is_cell = False
                bitarr = self._global_var_snp_bitarr_dict[chrom]
            snp_idxs = np.array([]).astype("int")
            for (start, end) in read.blocks:
                block_snp_bitarray = bitarr[start : end + 1]
                snp_idxs = np.append(snp_idxs, np.where(block_snp_bitarray.tolist())[0] + start)
            if is_cell:
                cell_snp_reads.append((read, snp_idxs))
            else:
                soup_snp_reads.append((read, snp_idxs))
        return cell_snp_reads, soup_snp_reads

    @staticmethod
    def _get_bam_stats(bam_filepath):
        stats_str = StringIO(pysam.idxstats(bam_filepath))
        col_names = ["ref chrom", "ref len", "mapped", "unmapped"]
        bam_stats_df = pd.read_csv(stats_str, delimiter="\t", names=col_names, index_col="ref chrom")
        return bam_stats_df

    def _extract_snp_data(
        self, cell_snp_reads, soup_snp_reads, cell_id_donor_lookup
    ) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
        cell_nativity_lookup = {cell_id: {"native": 0, "foreign": 0} for cell_id in cell_id_donor_lookup.keys()}
        donor_soup_nativity_lookup = {donor: {"native": 0, "foreign": 0} for donor in self.donors}
        soup_var_snp_df, donor_cell_var_snp_df, soup_unq_snp_df, donor_cell_unq_snp_df = self._init_snp_dfs(
            self._global_ref_alleles, self.donors
        )
        umis_seen = dict()
        count = 0
        n_reads = self.bam_stats_df[["mapped", "unmapped"]].sum().sum()
        pbar_chunk = 1000
        pbar_size = math.ceil(n_reads / pbar_chunk)
        snp_reads = cell_snp_reads + soup_snp_reads
        with tqdm(total=pbar_size) as pbar:
            for (read, idxs) in snp_reads:
                count += 1
                if count % pbar_chunk == 0:
                    pbar.update(pbar_chunk)
                umi = read.get_tag("UB")
                if umi in umis_seen:
                    umis_seen[umi] += 1
                    continue
                umis_seen[umi] = 1
                chrom = read.reference_name
                drop_id = read.get_tag("CB")
                if drop_id in cell_nativity_lookup:
                    donor = cell_id_donor_lookup.get(drop_id, None)
                    umi_native = self._check_native_read(read, idxs, donor)
                    for pos in idxs:
                        ref_allele = self._global_ref_alleles[(chrom, pos)]
                        is_ref = self._check_pos_allele(read, pos, ref_allele)
                        allele_type = "ref" if is_ref else "alt"
                        donor_cell_var_snp_df[donor].loc[chrom, pos, allele_type] += 1
                    if umi_native is True:
                        cell_nativity_lookup[drop_id]["native"] += 1
                    elif umi_native is False:
                        cell_nativity_lookup[drop_id]["foreign"] += 1
                    elif umi_native is None:
                        continue
                else:
                    for donor, soup_nativity_lookup in donor_soup_nativity_lookup.items():
                        umi_native = self._check_native_read(read, idxs, donor)
                        for pos in idxs:
                            ref_allele = self._global_ref_alleles[(chrom, pos)]
                            is_ref = self._check_pos_allele(read, pos, ref_allele)
                            allele_type = "ref" if is_ref else "alt"
                            soup_var_snp_df[allele_type].loc[chrom, pos] += 1
                        if umi_native is True:
                            soup_nativity_lookup["native"] += 1
                        elif umi_native is False:
                            soup_nativity_lookup["foreign"] += 1
                        elif umi_native is None:
                            continue

        return cell_nativity_lookup, donor_soup_nativity_lookup, soup_var_snp_df, donor_cell_var_snp_df

    @staticmethod
    def _init_snp_dfs(global_ref_alleles, donors):
        multi_index_tuples = [
            tuple(y) for x in list(map(list, global_ref_alleles.index)) for y in (x + ["ref"], x + ["alt"])
        ]
        multi_index = pd.MultiIndex.from_tuples(multi_index_tuples, names=["chrom", "pos", "allele"])
        soup_snp_df = pd.DataFrame(0, index=global_ref_alleles.keys(), columns=["ref", "alt"]).astype("uint32")
        donor_cell_snp_df = pd.DataFrame(0, index=multi_index, columns=donors).astype("uint32")
        return soup_snp_df, donor_cell_snp_df

    def _check_native_read(self, read, idxs, donor):
        chrom = read.reference_name
        snp_seen = False
        allele_dict = self._donor_var_allele_dict[donor]
        read_matches = []
        for pos in idxs:
            if (chrom, pos) not in allele_dict:
                continue
            snp_seen = True
            allele = allele_dict[(chrom, pos)]
            read_matches.append(self._check_pos_allele(read, pos, allele))
        if not all(read_matches):
            return False
        if snp_seen:
            return True
        else:
            return None

    @staticmethod
    def _check_pos_allele(read, pos, allele):
        for block in read.blocks:
            if block[0] <= pos <= block[1]:
                read_idx = int(pos - block[0] - 1)
                read_allele = read.seq[read_idx]
                if read_allele != allele:
                    return False
        return True

    @staticmethod
    def estimate_donor_norm_factors(soup_snp_df: pd.DataFrame, donor_cell_snp_df: pd.DataFrame):
        cell_soup_snp_mat = np.concatenate(
            [donor_cell_snp_df[donor_cell_snp_df.index.get_level_values("allele") == x] for x in ["ref", "alt"]], axis=0
        )
        cell_snp_mat = soup_snp_df.values.flatten("F")
        model = LinearRegression().fit(cell_soup_snp_mat, cell_snp_mat)
        donor_soup_fracs = pd.Series(model.coef_, index=donor_cell_snp_df.columns)
        return donor_soup_fracs

    @staticmethod
    def _build_cell_df(cell_id_donor_lookup, cell_nativity_lookup, cell_ids, counts):
        # TODO figure out contamination confidence
        # TODO: cases where cell nativity counts are too low -- use average value for whole lane
        cell_df_columns = ["donor", "native", "foreign"]
        cell_df_dict = {
            cell_id: [
                cell_id_donor_lookup[cell_id],
                cell_nativity_lookup[cell_id]["native"],
                cell_nativity_lookup[cell_id]["foreign"],
            ]
            for cell_id in cell_ids
        }
        cell_df = pd.DataFrame.from_dict(cell_df_dict, orient="index", columns=cell_df_columns)
        cell_df["total"] = cell_df["native"] + cell_df["foreign"]
        cell_df["counts"] = counts.sum(axis=1).reindex(cell_df.index)
        cell_df["foreign_frac"] = cell_df["foreign"] / cell_df["total"]
        cell_df["foreign_counts"] = cell_df["counts"] * cell_df["foreign_frac"]
        return cell_df

    @staticmethod
    def _build_donor_df(donor_soup_nativity_lookup, donors):
        donor_df_columns = ["native", "foreign"]
        donor_df_dict = {
            donor: [donor_soup_nativity_lookup[donor]["native"], donor_soup_nativity_lookup[donor]["foreign"]]
            for donor in donors
        }
        donor_df = pd.DataFrame.from_dict(donor_df_dict, orient="index", columns=donor_df_columns)
        donor_df["total"] = donor_df["native"] + donor_df["foreign"]
        donor_df["foreign_frac"] = donor_df["foreign"] / donor_df["total"]
        return donor_df

    @staticmethod
    def remove_bad_donors(cell_df, donor_df, donor_cell_snp_df, min_cells_per_donor):
        cell_df = cell_df.copy()
        donor_df = donor_df.copy()
        donor_cell_snp_df = donor_cell_snp_df.copy()
        donor_df["cells"] = cell_df.groupby("donor")["donor"].agg(np.size)
        donor_df = donor_df[donor_df["cells"] >= min_cells_per_donor]
        donors = list(donor_df.index)
        cell_df = cell_df[cell_df["donor"].isin(donors)]
        donor_cell_snp_df = donor_cell_snp_df[donors]
        return cell_df, donor_df, donor_cell_snp_df, donors

    @staticmethod
    def estimate_contamination(cell_df, donor_df):
        """ """
        cell_df["contamination_frac"] = cell_df["foreign_frac"] / cell_df["donor"].apply(
            lambda x: donor_df["foreign_frac"][x]
        )
        cell_df["contaminated_counts"] = cell_df["counts"] * cell_df["contamination_frac"]
        return cell_df

    @staticmethod
    def correct_expression(counts, cell_df, soup_expr_prof, mode="conservative"):
        # TODO: check that soup_expr vector is in same order as counts
        soup_expr_prof = soup_expr_prof.reindex(list(counts.columns))
        if mode == "conservative":
            cell_bad_counts = cell_df["foreign_counts"].reindex(counts.index)
        elif mode == "normalized":
            cell_bad_counts = cell_df["contaminated_counts"].reindex(counts.index)
        counts_contamination = pd.DataFrame(
            np.outer(cell_bad_counts, soup_expr_prof), index=cell_bad_counts.index, columns=soup_expr_prof.index
        )
        assert all(counts.index == counts_contamination.index) and all(counts.columns == counts_contamination.columns)
        corrected_counts = counts - counts_contamination
        return corrected_counts

    def save(self, save_dir, save_prefix):
        state = ChowderIO()
        public_attrs = self._filter_attrs(self)
        for attr_name in public_attrs:
            setattr(state, attr_name, getattr(self, attr_name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filepath = Path(save_dir) / (save_prefix + f"_{self.batch}_{self.lane}.pickle")
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath):
        batch, lane = filepath.split("/")[-1].split(".")[-2].split("_")[-2:]
        inst = cls(batch, lane)
        with open(filepath, "rb") as f:
            save_state = pickle.load(f)
        public_attrs = cls._filter_attrs(save_state)
        for attr_name in public_attrs:
            setattr(inst, attr_name, getattr(save_state, attr_name))
        inst.chowderx_save = filepath
        return inst

    @staticmethod
    def _filter_attrs(obj):
        public_attrs = []
        for attr_name in dir(obj):
            if (not attr_name.startswith("_")) and (not callable(getattr(obj, attr_name))):
                public_attrs.append(attr_name)
        return public_attrs

    def calc_droplet_stats(self):
        n_cells = self.counts.shape[0]
        n_droplets = self.counts_raw.shape[0]
        n_cell_umis = self.counts.sum().sum()
        n_umis = self.counts_raw.sum()
        index = ["n_cells", "n_droplets", "n_cell_umis", "n_umis"]
        droplet_stats = pd.Series([n_cells, n_droplets, n_cell_umis, n_umis], index=index)
        droplet_stats["frac_umis_cellular"] = n_cell_umis / n_umis
        droplet_stats["frac_drops_contain_cells"] = n_cells / n_droplets
        return droplet_stats

    def get_soup_counts(self):
        soup_barcodes = set(self.drop_ids["barcode"]).difference(self.counts.index)
        soup_indices = self.drop_ids[self.drop_ids["barcode"].isin(soup_barcodes)].index
        soup_counts = self.counts_raw.tocsr()[soup_indices]
        return soup_barcodes, soup_counts
