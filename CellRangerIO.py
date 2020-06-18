import gzip
import logging
import os

import pandas as pd
from scipy import io, sparse

logger = logging.getLogger()


class CellRangerIO:
    FEATURES_FILE_NAME = "features.tsv.gz"
    BARCODES_FILE_NAME = "barcodes.tsv.gz"
    MATRIX_UNZIPPED_FILE_NAME = "matrix.mtx"
    MATRIX_FILE_NAME = "matrix.mtx.gz"

    @staticmethod
    def load_dir(dir_, sparse=False, merged=True):
        if sparse and merged:
            logger.warning("`merged=True` not compatible with `sparse=True`. Setting `merged=False`")
            merged = False
        fmt = "sparse" if sparse else "dense"
        features_path, barcodes_path, matrix_path = CellRangerIO._get_10x_paths(dir_)
        logger.info(f"loading 10X data from {dir_} in {fmt} format...")

        features_df = pd.read_csv(features_path, compression="gzip", header=None, sep="\t", usecols=[0, 1]).rename(
            columns={0: "ensembl_id", 1: "gene_name"}
        )
        barcodes_df = pd.read_csv(barcodes_path, compression="gzip", header=None, sep="\t").rename(
            columns={0: "barcode"}
        )
        barcodes_df = pd.DataFrame(barcodes_df)
        if sparse:
            with gzip.open(matrix_path) as f:
                mat = io.mmread(f).T.tocsr()
        else:
            mat = pd.read_csv(
                matrix_path, skiprows=3, header=None, sep=" ", names=["feature_idx", "barcode_idx", "count"]
            )
        if not merged:
            return features_df, barcodes_df, mat
        features_df["feature_idx"] = features_df.index
        barcodes_df["barcode_idx"] = barcodes_df.index
        mat.barcode_idx -= 1  # convert from Matrix Market
        mat.feature_idx -= 1
        mat = mat.merge(features_df, on="feature_idx")
        mat = mat.merge(barcodes_df, on="barcode_idx")
        counts = mat.pivot_table(index="barcode", columns="gene_name", values="count", aggfunc="sum")
        counts.fillna(0, inplace=True)
        counts = counts.astype("uint16")
        return counts

    @staticmethod
    def save_dir(counts, output_dir):
        logger.info(f"saving counts data to {output_dir}...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        features_path, barcodes_path, matrix_path = CellRangerIO._get_10x_paths(output_dir)
        matrix_tmp_path = os.path.join(output_dir, "matrix_tmp.mtx")
        features_df = pd.DataFrame({0: counts.columns, 1: counts.columns, 2: "Gene Expresssion"})
        barcodes_df = pd.Series(counts.index)
        matrix = sparse.csr_matrix(counts.values).T

        barcodes_df["barcode_idx"] = barcodes_df.index + 1
        features_df["feature_idx"] = features_df.index + 1
        features_df.to_csv(features_path, header=False, sep="\t", index=False, compression="gzip")
        barcodes_df.to_csv(barcodes_path, header=False, sep="\t", index=False, compression="gzip")
        io.mmwrite(matrix_tmp_path, matrix, field="integer")
        matrix_df = pd.read_csv(
            matrix_tmp_path, skiprows=3, header=None, sep=" ", names=["feature_idx", "barcode_idx", "count"]
        ).astype("int")

        header = "%%MatrixMarket matrix coordinate integer general\n"
        header += """%metadata_json: {"format_version": 2, "software_version": "3.0.2"}\n"""
        header += "\t".join(map(str, [len(features_df), len(barcodes_df), matrix.count_nonzero()]))
        header += "\n"
        with open(matrix_path, "w") as f:
            f.write(header)
            matrix_df[["feature_idx", "barcode_idx", "count"]].to_csv(
                f, index=False, sep=" ", header=False, compression="gzip"
            )
        # with open(matrix_unzipped_path, 'rb') as f_in, gzip.open(matrix_path, 'wb') as f_out:
        #     copyfileobj(f_in, f_out)

    @staticmethod
    def save_from_sparse(features_df, barcodes_df, matrix, output_dir):
        pass

    @staticmethod
    def _get_10x_paths(dir_):
        features_path = os.path.join(dir_, CellRangerIO.FEATURES_FILE_NAME)
        barcodes_path = os.path.join(dir_, CellRangerIO.BARCODES_FILE_NAME)
        matrix_path = os.path.join(dir_, CellRangerIO.MATRIX_FILE_NAME)
        return features_path, barcodes_path, matrix_path
