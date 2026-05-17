import narwhals as nw
import narwhals.typing as nwt


def narwhals_df_equals(l1: nwt.DataFrameT, l2: nwt.DataFrameT) -> bool:
    """Determines equality of two DataFrames. Considers them equal if they have the same type,
    columns and rows, with ordering for both being the same as well."""
    # Two DataFrames are not equal if they have different types, different columns, or a different
    # number of rows.
    if type(l1) is not type(l2) or l1.columns != l2.columns or len(l1) != len(l2):
        return False

    # Otherwise, they are equal iff an outer join on all columns including row index creates no
    # extra rows.
    assert "__index" not in l1.columns
    return len(l1) == (
        l1
        .with_row_index("__index")
        .lazy()
        .join(
            l2.with_row_index("__index").lazy(), on=[*l1.columns, "__index"], how="full"
        )
        .select(nw.col("__index").fill_null(-1).count())
        .collect()
        .item()
    )
