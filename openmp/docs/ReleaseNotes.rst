===========================
OpenMP 12.0.0 Release Notes
===========================


.. warning::
   These are in-progress notes for the upcoming LLVM 12.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the OpenMP runtime, release 12.0.0.
Here we describe the status of OpenMP, including major improvements
from the previous release. All OpenMP releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

- Extended the ``libomptarget`` API functions to include source location
  information and OpenMP target mapper support. This allows ``libomptarget`` to
  know the source location of the OpenMP region it is executing, as well as the
  name and declarations of all the variables used inside the region. Each
  function generated now uses its ``mapper`` variant. The old API calls now call
  into the new API functions with ``nullptr`` arguments for backwards
  compatibility with old binaries. Source location information for
  ``libomptarget`` is now generated by Clang at any level of debugging
  information.

- Added improved error messages for ``libomptarget`` and ``CUDA`` plugins. Error
  messages are now presented without requiring a debug build of
  ``libomptarget``. The newly added source location information can also be used
  to identify which OpenMP target region the failure occurred in. More
  information can be found :ref:`here <libopenmptarget_errors>`.

- Added additional environment variables to control output from the
  ``libomptarget`` runtime library. ``LIBOMPTARGET_PROFILE`` to
  generate time profile output similar to Clang's ``-ftime-trace`` option.
  ``LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD`` sets the threshold size for which
  the ``libomptarget`` memory manager will handle the allocation.
  ``LIBOMPTARGET_INFO`` allows the user to request certain information from the
  ``libomptarget`` runtime using a 32-bit field. A full description of each
  environment variable is described :ref:`here <libopenmptarget_environment_vars>`.

- ``target nowait`` was supported via hidden helper task, which is a task not
  bound to any parallel region. A hidden helper team with a number of threads is
  created when the first hidden helper task is encountered. The number of threads
  can be configured via the environment variable
  ``LIBOMP_NUM_HIDDEN_HELPER_THREADS``. By default it is 8. If
  ``LIBOMP_NUM_HIDDEN_HELPER_THREADS=0``, hidden helper task is disabled and
  falls back to a regular OpenMP task. It can also be disabled by setting the
  environment variable ``LIBOMP_USE_HIDDEN_HELPER_TASK=OFF``.

- ``deviceRTLs`` for NVPTX platform is CUDA free now. It is generally OpenMP code.
  Target dependent parts are implemented with Clang/LLVM/NVVM intrinsics. CUDA
  SDK is also dropped as a dependence to build the device runtime, which means
  device runtime can also be built on a CUDA free system. However, it is
  disabled by default. Set the CMake variable
  ``LIBOMPTARGET_BUILD_NVPTX_BCLIB=ON`` to enable the build of NVPTX device
  runtime on a CUDA free system. ``gcc-multilib`` and ``g++-multilib`` are
  required. If CUDA is found, the device runtime will be built by default.

- Static NVPTX device runtime library (``libomptarget-nvptx.a``) was dropped.
  A bitcode library is required to build an OpenMP program. If the library is
  not found in the default path or any of the paths defined by ``LIBRARY_PATH``,
  an error will be raised. User can also specify the path to the bitcode device
  library via ``--libomptarget-nvptx-bc-path=``.
