# noinspection PyPep8Naming
def NamedIndexSlice(**kwargs) -> tuple:
    """Kwargs based index slice: name the index levels and specify the slices

    Args:
        **kwargs: level to slice object mapping

    Returns:
        function accepting a dataframe as input and returning an indexing
        tuple, for use in .loc indexing

    Examples:
        nidxs = NamedIndexSlice
        df.loc[nidxs(level1='a', level2=[1, 2], level5=[True, True, False]), :]

    Note:
        This function uses the ability of the .loc indexing to accept a
        callable. This callable is called with the dataframe as only
        argument and must return an object that can be evaluated by loc.

        This function returns a function which will produce a tuple with
        correct assignment of slice object to level, inserting slice(None)
        where necessary.

        Consecutive trailing slice(None) objects are removed from the
        indexing tuple. This is done to deal with issue
        https://github.com/pandas-dev/pandas/issues/18631 which essentially
        means: indexing with one or more scalars, e.g. ('a', ) or ('a', 'b')
        -> index levels are dropped. Indexing with one or more scalars
        followed by slice(None) or a list of values -> the index levels are
        not dropped. By leaving out trailing slice(None) objects index
        levels are dropped for scalar only indexing, which is the same
        behavior one would have achieved with standard loc indexing (where
        usually trailing slice(None) or : specs (in IndexSlice) are omitted
    """

    def fn(df):
        slicing_list = [kwargs.get(index_name, slice(None))
                        for index_name in df.index.names]
        for i in reversed(range(len(slicing_list))):
            if slicing_list[i] == slice(None):
                slicing_list.pop(i)
            else:
                break

        return tuple(slicing_list)

    return fn

def NamedColumnsSlice(**kwargs) -> tuple:
    """Kwargs based index slice: name the index levels and specify the slices

    Args:
        **kwargs: level to slice object mapping

    Returns:
        function accepting a dataframe as input and returning an indexing
        tuple, for use in .loc indexing

    Examples:
        nidxs = NamedIndexSlice
        df.loc[nidxs(level1='a', level2=[1, 2], level5=[True, True, False]), :]

    Note:
        This function uses the ability of the .loc indexing to accept a
        callable. This callable is called with the dataframe as only
        argument and must return an object that can be evaluated by loc.

        This function returns a function which will produce a tuple with
        correct assignment of slice object to level, inserting slice(None)
        where necessary.

        Consecutive trailing slice(None) objects are removed from the
        indexing tuple. This is done to deal with issue
        https://github.com/pandas-dev/pandas/issues/18631 which essentially
        means: indexing with one or more scalars, e.g. ('a', ) or ('a', 'b')
        -> index levels are dropped. Indexing with one or more scalars
        followed by slice(None) or a list of values -> the index levels are
        not dropped. By leaving out trailing slice(None) objects index
        levels are dropped for scalar only indexing, which is the same
        behavior one would have achieved with standard loc indexing (where
        usually trailing slice(None) or : specs (in IndexSlice) are omitted
    """

    def fn(df):
        slicing_list = [kwargs.get(index_name, slice(None))
                        for index_name in df.columns.names]
        for i in reversed(range(len(slicing_list))):
            if slicing_list[i] == slice(None):
                slicing_list.pop(i)
            else:
                break

        print('using ', tuple(slicing_list))

        return tuple(slicing_list)

    return fn
