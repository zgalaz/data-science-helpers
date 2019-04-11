def starify_pval(pval):
    if pval > 0.05:
        return ""
    else:
        if pval <= 0.001:
            return "***"
        if pval <= 0.01:
            return "**"
        if pval <= 0.05:
            return "*"
