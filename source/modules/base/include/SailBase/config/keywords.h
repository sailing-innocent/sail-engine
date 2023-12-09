#pragma once

#ifndef SAIL_IMPORT
#  if defined(_MSC_VER)
#    define SAIL_IMPORT __declspec(dllimport)
#  else
#    define SAIL_IMPORT __attribute__((visibility("default")))
#  endif
#endif

#ifndef SAIL_EXPORT
#  if defined(_MSC_VER)
#    define SAIL_EXPORT __declspec(dllexport)
#  else
#    define SAIL_EXPORT __attribute__((visibility("default")))
#  endif
#endif