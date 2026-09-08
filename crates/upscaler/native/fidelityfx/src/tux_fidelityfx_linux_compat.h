#ifndef TUX_FIDELITYFX_LINUX_COMPAT_H
#define TUX_FIDELITYFX_LINUX_COMPAT_H

#if !defined(_WIN32)

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <new>

#ifndef _countof
#define _countof(array) (sizeof(array) / sizeof((array)[0]))
#endif

#ifndef FFX_UNUSED
#define FFX_UNUSED(value) ((void)(value))
#endif

namespace tux_fidelityfx {

inline void utf8_to_wide(const char* input, wchar_t* output, std::size_t output_size)
{
    if (!output || output_size == 0) {
        return;
    }
    output[0] = L'\0';
    if (!input) {
        return;
    }

    std::size_t written = 0;
    for (std::size_t index = 0; input[index] != '\0' && written + 1 < output_size;) {
        const auto lead = static_cast<unsigned char>(input[index++]);
        std::uint32_t codepoint = 0xfffd;
        std::size_t continuation_count = 0;
        if (lead < 0x80) {
            codepoint = lead;
        } else if ((lead & 0xe0) == 0xc0) {
            codepoint = lead & 0x1f;
            continuation_count = 1;
        } else if ((lead & 0xf0) == 0xe0) {
            codepoint = lead & 0x0f;
            continuation_count = 2;
        } else if ((lead & 0xf8) == 0xf0) {
            codepoint = lead & 0x07;
            continuation_count = 3;
        }

        bool valid = continuation_count != 0 || lead < 0x80;
        for (std::size_t continuation = 0; valid && continuation < continuation_count; ++continuation) {
            const auto byte = static_cast<unsigned char>(input[index]);
            if ((byte & 0xc0) != 0x80) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6) | (byte & 0x3f);
            ++index;
        }
        if (!valid || codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff) ||
            (continuation_count == 1 && codepoint < 0x80) ||
            (continuation_count == 2 && codepoint < 0x800) ||
            (continuation_count == 3 && codepoint < 0x10000)) {
            codepoint = 0xfffd;
        }

        if (sizeof(wchar_t) >= 4 || codepoint <= 0xffff) {
            output[written++] = static_cast<wchar_t>(codepoint);
        } else if (written + 2 < output_size) {
            codepoint -= 0x10000;
            output[written++] = static_cast<wchar_t>(0xd800 | (codepoint >> 10));
            output[written++] = static_cast<wchar_t>(0xdc00 | (codepoint & 0x3ff));
        } else {
            break;
        }
    }
    output[written] = L'\0';
}

inline void wide_to_utf8(const wchar_t* input, char* output, std::size_t output_size)
{
    if (!output || output_size == 0) {
        return;
    }
    output[0] = '\0';
    if (!input) {
        return;
    }

    std::size_t written = 0;
    for (std::size_t index = 0; input[index] != L'\0'; ++index) {
        std::uint32_t codepoint = static_cast<std::uint32_t>(input[index]);
        if (sizeof(wchar_t) == 2 && codepoint >= 0xd800 && codepoint <= 0xdbff &&
            input[index + 1] >= 0xdc00 && input[index + 1] <= 0xdfff) {
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) +
                (static_cast<std::uint32_t>(input[++index]) - 0xdc00);
        }
        if (codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            codepoint = 0xfffd;
        }

        char encoded[4];
        std::size_t encoded_size = 0;
        if (codepoint < 0x80) {
            encoded[0] = static_cast<char>(codepoint);
            encoded_size = 1;
        } else if (codepoint < 0x800) {
            encoded[0] = static_cast<char>(0xc0 | (codepoint >> 6));
            encoded[1] = static_cast<char>(0x80 | (codepoint & 0x3f));
            encoded_size = 2;
        } else if (codepoint < 0x10000) {
            encoded[0] = static_cast<char>(0xe0 | (codepoint >> 12));
            encoded[1] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
            encoded[2] = static_cast<char>(0x80 | (codepoint & 0x3f));
            encoded_size = 3;
        } else {
            encoded[0] = static_cast<char>(0xf0 | (codepoint >> 18));
            encoded[1] = static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
            encoded[2] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
            encoded[3] = static_cast<char>(0x80 | (codepoint & 0x3f));
            encoded_size = 4;
        }
        if (written + encoded_size >= output_size) {
            break;
        }
        std::memcpy(output + written, encoded, encoded_size);
        written += encoded_size;
    }
    output[written] = '\0';
}

} // namespace tux_fidelityfx

inline int strcpy_s(char* destination, std::size_t destination_size, const char* source)
{
    if (!destination || destination_size == 0) {
        return 1;
    }
    if (!source) {
        destination[0] = '\0';
        return 1;
    }
    std::snprintf(destination, destination_size, "%s", source);
    return 0;
}

inline int wcscpy_s(wchar_t* destination, std::size_t destination_size, const wchar_t* source)
{
    if (!destination || destination_size == 0) {
        return 1;
    }
    if (!source) {
        destination[0] = L'\0';
        return 1;
    }
    std::wcsncpy(destination, source, destination_size - 1);
    destination[destination_size - 1] = L'\0';
    return 0;
}

template <std::size_t Size>
inline int wcscpy_s(wchar_t (&destination)[Size], const wchar_t* source)
{
    return wcscpy_s(destination, Size, source);
}

#define sprintf_s(buffer, size, ...) std::snprintf(buffer, size, __VA_ARGS__)
#define swprintf_s(buffer, size, ...) std::swprintf(buffer, size, __VA_ARGS__)

#endif

#endif
