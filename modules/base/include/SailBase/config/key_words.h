#pragma once
//-------------------------------------------------------------------------------
// unused
// alignas
// assume
// enable/disable optimization
// inline
// forceinline
// extern c
// export/import/static API
// ptr size
// no vtable
// noexcept
//-------------------------------------------------------------------------------

// INLINE
#if defined(__cplusplus)
#define SAIL_INLINE inline
#else
#define SAIL_INLINE
#endif

// IMPORT
#ifndef SAIL_IMPORT
#if defined(_MSC_VER)
#define SAIL_IMPORT __declspec(dllimport)
#else
#define SAIL_IMPORT __attribute__((visibility("default")))
#endif
#endif

// EXPORT
#ifndef SAIL_EXPORT
#if defined(_MSC_VER)
// MSVC linker trims symbols, the 'dllexport' attribute prevents this.
// But we are not archiving DLL files with SHIPPING_ONE_ARCHIVE mode.
#define SAIL_EXPORT __declspec(dllexport)
#else
#define SAIL_EXPORT __attribute__((visibility("default")))
#endif
#endif
