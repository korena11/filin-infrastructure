# This is free and unencumbered software released into the public domain.
# See UNLICENSE.

import contextlib

import numpy as np
import numpy.core.arrayprint as arrayprint


#
# cc-wiki
# http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
#
@contextlib.contextmanager
def printoptions(strip_zeros=True, **kwargs):
    origcall = arrayprint.FloatFormat.__call__

    # noinspection PyUnresolvedReferences
    def __call__(self, x, strip_zeros=strip_zeros):
        return origcall.__call__(self, x, strip_zeros)

    arrayprint.FloatFormat.__call__ = __call__
    original = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield
    np.set_printoptions(**original)
    arrayprint.FloatFormat.__call__ = origcall


# noinspection PyTypeChecker
def to_latex(a, decimals=4, tab='  '):
    """
    Convert an array-like object into a LaTeX array.
    The elements of each column are aligned on the decimal, if present.
    Spacing is automatically adjusted depending on if `a` contains negative
    numbers or not. For float data, trailing zeros are included so that
    array output is uniform.
    Parameters
    ----------
    a : array-like
        A list, tuple, NumPy array, etc. The elements are written into a
        LaTeX array environment.
    decimals : int
        The number of decimal places to round to before writing to LaTeX.
    tab : str
        The tab character to use for indentation within LaTeX.
    Examples
    --------
    >>> x = [1,2,-3]
    >>> print to_latex(x)
    \newcolumntype{X}{D{.}{.}{2,0}}
    \begin{array}{*{3}{X}}
      1 & 2 & -3
    \end{array}
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> y = np.random.rand(12).reshape(4,3) - 0.5
    >>> print to_latex(y, decimals=2)
    \begin{array}{*{3}{X}}
      0.05 & 0.22 & 0.10 \\
      0.04 & -0.08 & 0.15 \\
      -0.06 & 0.39 & 0.46 \\
      -0.12 & 0.29 & 0.03
    \end{array}
    Notes
    -----
    The resultant array environment should be used within LaTeX's mathmode,
    and the following should appear in the preamble:
        \\usepackage{array}
        \\usepackage{dcolumn}
    """
    # array = np.around(a, decimals)
    array = np.atleast_2d(a)

    # Determine the number of digits left of the decimal.
    # Grab integral part, convert to string, and get maximum length.
    # This does not handle negative signs appropriately since -0.5 goes to 0.
    # So we make sure it never counts a negative sign by taking the abs().
    integral = np.abs(np.trunc(array.flat).astype(int))
    left_digits = max(list(map(len, list(map(str, integral)))))

    # Adjust for negative signs.
    if np.any(array < 0):
        left_digits += 1

    # Set decimal digits to 0 if data are not floats.
    try:
        np.finfo(array.dtype)
    except ValueError:
        decimals = 0

    # Align the elements on the decimal, making room for largest element.

    coltype = r"\newcolumntype{{X}}{{D{{.}}{{.}}{{{0},{1}}}}}"
    coltype = coltype.format(left_digits, decimals)

    # Specify that we want all columns to have the same column type.
    nCols = array.shape[1]
    cols = r"*{{{nCols}}}{{X}}".format(nCols=nCols)

    # Build the lines in the array.
    #
    # In general, we could just use the rounded array and map(str, row),
    # but NumPy strips trailing zeros on floats (undesirably). So we make
    # use of the context manager to prevent that.
    options = {
        'precision': decimals,
        'suppress': True,
        'strip_zeros': False,
        'threshold': nCols + 1,
    }
    with printoptions(**options):
        lines = []
        for row in array:
            # Strip [ and ], remove newlines, and split on whitespace
            elements = row.__str__()[1:-1].replace('\n', '').split()
            line = [tab, ' & '.join(elements), r' \\']
            lines.append(''.join(line))

    # Remove the \\ on the last line.
    lines[-1] = lines[-1][:-3]

    # Create the LaTeX code
    subs = {'coltype': coltype, 'cols': cols, 'lines': '\n'.join(lines)}
    template = r"""
    {coltype}
\begin{{array}}{{{cols}}}
{lines}
\end{{array}}
"""

    return template.format(**subs)


# noinspection PyTypeChecker
def matrix_latex(a, **kwargs):
    """

    :param a: matrix to print
    :param name: name of the matrix (string)
    :param units: its units (string)
    :param decimals: how many numbers after decimal
    :param tab: break between columns
    :param suppress: suppress writing scientific (with e-10)
    :return: string with the matrix/vector in latex

    :USAGE:
    matrix_latex(A, name='{\bf A}', units='[m,m,-]', decimals=3, suppress=TRUE)
    """

    options = {'name': ' ',
               'units': ' ',
               'decimals': 4,
               'tab': '  ',
               'suppress': False
               }
    options.update(kwargs)
    name = options['name']
    units = options['units']
    decimals = options['decimals']
    tab = options['tab']
    suppress = options['suppress']

    # In general, we could just use the rounded array and map(str, row),
    # but NumPy strips trailing zeros on floats (undesirably). So we make
    # use of the context manager to prevent that.
    options = {
        'precision': decimals,
        'suppress': suppress,
        'strip_zeros': False,
    }
    if np.size(a.shape) == 1:
        column_vector = True
        row_size = a.shape[0]
        column_size = 1
    else:
        column_vector = False
        row_size, column_size = a.shape
    array = np.atleast_2d(a)

    row_size = int(row_size)
    column_size = int(column_size)
    size_str = r'%i \times %i' % (row_size, column_size)

    # Determine the number of digits left of the decimal.
    # Grab integral part, convert to string, and get maximum length.
    integral = np.abs(np.trunc(array.flat).astype(int))
    left_digits = max(list(map(len, list(map(str, integral)))))

    # Adjust for negative signs.
    if np.any(array < 0):
        left_digits += 1

    # Set decimal digits to 0 if data are not floats.
    try:
        np.finfo(array.dtype)
    except ValueError:
        decimals = 0

    matrixName = r"{name}_{{{size}}}".format(name=name, size=size_str)

    # Build the lines in the array.
    with printoptions(**options):
        lines = []
        for row in array:
            # Strip [ and ], remove newlines, and split on whitespace
            elements = row.__str__()[1:-1].replace('\n', '').split()
            if column_vector:
                line = [tab, r' \\ '.join(elements), r' \\']
            else:
                line = [tab, ' & '.join(elements), r' \\']
            lines.append(''.join(line))

    # Remove the \\ on the last line.
    lines[-1] = lines[-1][:-3]

    # Create the LaTeX code
    subs = {'matrixName': matrixName, 'lines': '\n'.join(lines), 'units': units}
    template = r"""\begin{{equation*}} {matrixName} =
    \begin{{bmatrix}}
    {lines}
        \end{{bmatrix}} _{{{units}}}
    \end{{equation*}}
    """

    return template.format(**subs)
