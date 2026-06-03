median = np.nanmedian(col)
col[mask] = median if not np.isnan(median) else 0.0
