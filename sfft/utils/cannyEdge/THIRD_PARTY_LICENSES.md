# Third-Party License Notices

Except for `__init__.py` and this notice file, the source files in this
directory contain code copied, adapted, or derived from scikit-image. They were
compared against the scikit-image 0.18.3 source files listed below.

The corresponding upstream files are:

- `_dtype.py`, from `skimage/util/dtype.py`
- `_gaussian.py`, from `skimage/filters/_gaussian.py`
- `_utils.py`, from `skimage/_shared/utils.py`
- `_warnings.py`, from `skimage/_shared/_warnings.py`
- `canny.py`, from `skimage/feature/_canny.py`

Upstream project: https://github.com/scikit-image/scikit-image

The copied, adapted, or derived scikit-image code is redistributed under the
BSD 3-Clause License below. Local files may include vendoring changes such as
renamed files, adjusted imports, formatting differences, or reduced subsets of
larger upstream modules.

## scikit-image BSD 3-Clause License

Copyright (C) 2019, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of skimage nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Additional Notice in `canny.py`

The copied Canny implementation also includes this upstream notice:

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
