from polder.eager._narwhals_df_equals import narwhals_df_equals
from polder.protocols.array import LabelFrameType


class Labels(tuple[LabelFrameType | None, ...]):
    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return False

        for l1, l2 in zip(self, other, strict=True):
            if l1 is None and l2 is None:
                continue

            if l1 is None or l2 is None:
                return False

            if not narwhals_df_equals(l1, l2):
                return False

        return True

    def __hash__(self) -> int:
        # No good hash function yet, for now we use a very simple hash one
        # with a lot of collisions.
        def hash_labels(axis_labels: LabelFrameType | None):
            if axis_labels is None:
                return None

            return (axis_labels.columns, len(axis_labels))

        hash_proxy = tuple(hash_labels(axis_labels) for axis_labels in self)

        return hash(hash_proxy)
