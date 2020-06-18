from matplotlib import pyplot as plt


class ChowderPlotter:
    SCATTER_KWARGS = {
        "marker": "o",
        "linestyle": "",
        "ms": 3,
        "alpha": 0.05,
    }
    HIST_KWARGS = {"bins": 50, "alpha": 0.25}

    def __init__(self, df):
        self.df = df

    def multi_scatter(
            self, df, x_var, y_var, grp_var, split_var, xlim=None, ylim=None, use_split_titles=True, **kwargs
    ):
        for name, sub_df in df.groupby(split_var):
            title = name if use_split_titles else None
            self.scatter(sub_df, x_var, y_var, grp_var, xlim, ylim, title, **kwargs)

    def scatter(self, df, x_var, y_var, grp_vars, xlim=None, ylim=None, title=None, figsize=(20, 10), **kwargs):
        kwargs = {**self.SCATTER_KWARGS.copy(), **kwargs}
        self.fig, self.ax = plt.subplots(figsize=figsize)
        df_grp = df.groupby(grp_vars)
        for name, group in df_grp:
            self.ax.plot(group[x_var], group[y_var], label=name, **kwargs)
        if title:
            self.ax.set_title(title)
        if xlim:
            self.ax.set_xlim(*xlim)
        if ylim:
            self.ax.set_ylim(*ylim)
        self.ax.set_xlabel(x_var)
        self.ax.set_ylabel(y_var)
        self.ax.legend()
        plt.show()

    def hist(self, df, var, grp_vars, xlim=None, ylim=None, title=None, figsize=(20, 10), **kwargs):
        kwargs = {**self.HIST_KWARGS.copy(), **kwargs}
        self.fig, self.ax = plt.subplots(figsize=figsize)
        df_grp = df.groupby(grp_vars)
        for name, group in df_grp:
            self.ax.hist(group[var], label=name, **kwargs)
        if title:
            self.ax.set_title(title)
        if xlim:
            self.ax.set_xlim(*xlim)
        if ylim:
            self.ax.set_ylim(*ylim)
        self.ax.set_xlabel(var)
        self.ax.legend()
        plt.show()

    def multi_hist(self, df, var, grp_var, split_var, xlim=None, ylim=None, use_split_titles=True, **kwargs):
        for name, sub_df in df.groupby(split_var):
            title = name if use_split_titles else None
            self.scatter(sub_df, var, grp_var, xlim, ylim, title, **kwargs)

    def _plot_droplet_stats(self, chow):
        pass
