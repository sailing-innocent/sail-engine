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

// EXTERN_C
#ifdef __cplusplus
#define SAIL_EXTERN_C extern "C"
#else
#define SAIL_EXTERN_C
#endif

#ifdef __cplusplus
#define SAIL_IF_CPP(...) __VA_ARGS__
#else
#define SAIL_IF_CPP(...)
#endif

// constexpr
#ifdef __cplusplus
#define SAIL_CONSTEXPR constexpr
#else
#define SAIL_CONSTEXPR const
#endif

// INLINE
#if defined(__cplusplus)
#define SAIL_INLINE inline
#else
#define SAIL_INLINE
#endif

// FORCEINLINE
#if defined(_MSC_VER) && !defined(__clang__)
#define SAIL_FORCEINLINE __forceinline
#else
#define SAIL_FORCEINLINE inline __attribute__((always_inline))
#endif

// ALIGNAS
#if defined(_MSC_VER)
#define SAIL_ALIGNAS(x) __declspec(align(x))
#else
#define SAIL_ALIGNAS(x) __attribute__((aligned(x)))
#endif