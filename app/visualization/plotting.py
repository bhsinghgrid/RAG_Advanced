import pandas as pd
import io
import base64

# Lazy import for matplotlib to avoid hard dependency during import-time
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def plot_from_csv_string(csv_str: str, x: str = None, y: str = None, kind: str = "bar") -> str:
    """
    Create a simple plot from a CSV string and return a base64-encoded PNG data URI.
    x: column to use for x-axis; if None uses index or first column.
    y: column to plot; if None tries to guess the first numeric column.
    kind: matplotlib kind (bar/line)
    """
    df = pd.read_csv(io.StringIO(csv_str))
    if df.empty:
        return None

    if x is None:
        x = df.columns[0]

    if y is None:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric:
            y = numeric[0]
        else:
            # No numeric columns to plot
            return None

    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment")

    fig, ax = plt.subplots(figsize=(6, 3))
    if kind == "bar":
        ax.bar(df[x].astype(str), df[y])
    else:
        ax.plot(df[x].astype(str), df[y], marker='o')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} by {x}")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"


if __name__ == "__main__":
    csv = "col,A,B\nrow1,10,20\nrow2,15,25"
    uri = plot_from_csv_string(csv, x='col', y='A')
    print(bool(uri))
