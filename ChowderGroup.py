import logging
import os
from glob import glob
from pathlib import Path

import pandas as pd

from chowderx.ChowderX import ChowderX

logger = logging.getLogger()


class ChowderGroup:
    def __init__(self, saves):
        self._chows = []
        if isinstance(saves, (str, Path)):
            saves = sorted(glob(os.path.join(saves, "*")))
        for save_file in saves:
            logger.info(f"loading ChowderX file {save_file}...")
            chow = ChowderX.load(save_file)
            if "lane" not in chow.cell_df.columns:
                chow.cell_df.insert(1, "lane", chow.lane)
                chow.donor_df.index = pd.MultiIndex.from_product([chow.donor_df.index, [chow.lane]])
                chow.bam_stats_df.index = pd.MultiIndex.from_product([chow.bam_stats_df.index, [chow.lane]])
                chow.cell_ids = {cell_id + "_" + str(chow.lane) for cell_id in chow.cell_ids}
                chow.counts.index += "_" + str(chow.lane)
            self._chows.append(chow)
        # self.file_loc_df = self._get_file_loc_df()
        logger.info(f"building master dataset...")
        self.lanes = self._get_group_attr("lane")
        self.donors = sorted(list(set().union(*tuple(self._get_group_attr("donors")))))
        self.cell_ids = set().union(*(chow.cell_ids for chow in self._chows))
        self.soup_ids = set().union(*(chow.soup_ids for chow in self._chows))
        self.counts = self._concatenate_group_dfs("counts")
        self.cell_df = self._concatenate_group_dfs("cell_df")
        self.donor_df = self._concatenate_group_dfs("donor_df")
        self.bam_stats_df = self._concatenate_group_dfs("bam_stats_df")
        self.donor_norm_factors = self._stack_group_series("donor_norm_factors")
        self.soup_expr = self._stack_group_series("soup_expr")
        self.expr_prof = self._stack_group_series("soup_expr_prof")
        self.counts.fillna(0)

    def _get_file_loc_df(self):
        df_dict = {}
        # TODO: add this in below: 'chowderx_save' (raising error)
        metadata_attrs = [
            "lane_path",
            "demuxlet_path",
            "tenx_path",
            "tenx_raw_path",
            "lane_path",
            "demuxlet_path",
            "tenx_path",
            "tenx_raw_path",
            "tenx_filtered_path",
            "raw_barcodes_path",
            "raw_features_path",
            "filtered_matrix_path",
            "_bam_path_remote",
            "_bai_path_remote",
            "_bam_local_dir",
            "_bam_local_path",
            "_bai_local_path",
        ]
        for attr_name in metadata_attrs:
            df_dict[attr_name] = self._get_group_attr(attr_name)
        df = pd.DataFrame.from_dict(df_dict, orient="index", columns=self.lanes)
        return df

    def _get_group_attr(self, attr_name):
        return [getattr(chow, attr_name) for chow in self._chows]

    def _concatenate_group_dfs(self, df_name):
        return pd.concat([getattr(chow, df_name) for chow in self._chows], sort=False)

    def _stack_group_series(self, s_name):
        df = pd.concat([getattr(chow, s_name) for chow in self._chows], axis=1, names=self.lanes, sort=False)
        df.columns = self.lanes
        return df

    # wrapper to contain multiple ChowderX instances for different lanes or different processing conditions and allow
    # seamless experimentation on all. Sort of like Pandas GroupBy with dataframes. Could have a staticmethod which is a
    # decorator that applies the function to each ChowderX in the data structure
