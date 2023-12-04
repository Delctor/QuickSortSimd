
#include <iostream>
#include <immintrin.h>
#include <bitset>

#define zeroI 0, 1  // 0
#define oneI 2, 3   // 1
#define twoI 4, 5   // 2
#define threeI 6, 7 // 3


static __m256i lookUpTableRightPermute[16]
{
    _mm256_setr_epi32(zeroI, oneI, twoI, threeI), // 0b0000
    _mm256_setr_epi32(oneI, twoI, threeI, zeroI), // 0b0001
    _mm256_setr_epi32(zeroI, twoI, threeI, oneI), // 0b0010
    _mm256_setr_epi32(twoI, threeI, oneI, zeroI), // 0b0011
    _mm256_setr_epi32(zeroI, oneI, threeI, twoI), // 0b0100
    _mm256_setr_epi32(oneI, threeI, twoI, zeroI), // 0b0101
    _mm256_setr_epi32(zeroI, threeI, oneI, twoI), // 0b0110
    _mm256_setr_epi32(threeI, zeroI, oneI, twoI), // 0b0111
    _mm256_setr_epi32(zeroI, oneI, twoI, threeI), // 0b1000
    _mm256_setr_epi32(oneI, twoI, zeroI, threeI), // 0b1001
    _mm256_setr_epi32(zeroI, twoI, oneI, threeI), // 0b1010
    _mm256_setr_epi32(twoI, zeroI, oneI, threeI), // 0b1011
    _mm256_setr_epi32(zeroI, oneI, twoI, threeI), // 0b1100
    _mm256_setr_epi32(oneI, zeroI, twoI, threeI), // 0b1101
    _mm256_setr_epi32(zeroI, oneI, twoI, threeI), // 0b1110
    _mm256_setr_epi32(zeroI, oneI, twoI, threeI), // 0b1111
};

static __m256i lookUpTableLeftPermute[16]
{
    _mm256_permute4x64_epi64(lookUpTableRightPermute[0], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[1], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[2], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[3], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[4], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[5], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[6], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[7], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[8], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[9], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[10], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[11], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[12], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[13], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[14], 0b00011011),
    _mm256_permute4x64_epi64(lookUpTableRightPermute[15], 0b00011011),
};

#define swapSimd(a, b, mask) \
__m256d tmp = _mm256_blendv_pd(b, a, mask); \
a = _mm256_blendv_pd(a, b, mask); \
b = tmp;

size_t partitionNormal(double* arr, size_t l, size_t r)
{
    size_t pivot = r;

    size_t i = l, j = l;

    for (; i < pivot; i++)
    {
        if (arr[i] < arr[pivot])
        {

            std::swap(arr[i], arr[j]);
            j++;
        }
    }
    std::swap(arr[j], arr[pivot]);
    return j;
}

size_t partitionSimd(double* arr, size_t l, size_t r)
{
    __m256d _pivot = _mm256_broadcast_sd(&arr[r]);

    __m256d _maskMinusOne = _mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF));

    __m256d _lastIterationElements = _mm256_undefined_pd(), _lastIterationMask = _mm256_undefined_pd();
    size_t index = l;

    int lastIterationMask = 0b0000, lastIterationCount = 0;

    size_t space = r - l;

    size_t finalPos = (space / 4) * 4;

    finalPos = ((finalPos >= 8) * (finalPos)) + l;

    if (space >= 8)
    {
        _lastIterationElements = _mm256_load_pd(&arr[l]);
        _lastIterationMask = _mm256_cmp_pd(_lastIterationElements, _pivot, _CMP_LT_OQ);
        lastIterationMask = _mm256_movemask_pd(_lastIterationMask);

        __m256i _indicesToPermute = lookUpTableLeftPermute[lastIterationMask];
        _lastIterationElements = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_lastIterationElements), _indicesToPermute));
        _lastIterationMask = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_lastIterationMask), _indicesToPermute));
        lastIterationMask = _mm256_movemask_pd(_lastIterationMask);
        lastIterationCount = _mm_popcnt_u32(lastIterationMask);

        for (size_t i = l + 4; i < finalPos; i += 4)
        {
            // Load data
            __m256d _data = _mm256_load_pd(&arr[i]);

            // Get mask
            __m256d _mask = _mm256_cmp_pd(_data, _pivot, _CMP_LT_OQ);

            // Move mask to int
            int mask = _mm256_movemask_pd(_mask);
            int maskCount = _mm_popcnt_u32(mask);

            // Get indices to perform permutation
            __m256i _indicesToPermute = lookUpTableRightPermute[mask];

            // Perform permutations
            _data = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_data), _indicesToPermute));
            _mask = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_mask), _indicesToPermute));

            // Get mask for blend
            __m256d _maskBlend = _mm256_and_pd(_mm256_xor_pd(_lastIterationMask, _maskMinusOne), _mask);

            // Swap
            swapSimd(_data, _lastIterationElements, _maskBlend);

            // Get new mask
            _lastIterationMask = _mm256_or_pd(_lastIterationMask, _mask);
            lastIterationMask = _mm256_movemask_pd(_lastIterationMask);

            // Get indices
            _indicesToPermute = lookUpTableLeftPermute[lastIterationMask];

            // Perform permutation
            _lastIterationElements = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_lastIterationElements), _indicesToPermute));
            _lastIterationMask = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_lastIterationMask), _indicesToPermute));

            // Get new Count
            lastIterationCount = _mm_popcnt_u32(lastIterationMask);

            // Store
            _mm256_store_pd(&arr[index], _lastIterationElements);

            // Remaining elements
            _mask = _mm256_xor_pd(_maskBlend, _mask);
            int mask2 = _mm256_movemask_pd(_mask);

            _indicesToPermute = lookUpTableLeftPermute[mask2];
            _mask = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_mask), _indicesToPermute));
            _data = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_data), _indicesToPermute));

            if (index + 4 < i && mask2)
            {
                index += 4;
                _lastIterationElements = _mm256_load_pd(&arr[index]);
                _lastIterationMask = _mask;
                lastIterationMask = _mm256_movemask_pd(_lastIterationMask);
                lastIterationCount = _mm_popcnt_u32(lastIterationMask);

                swapSimd(_lastIterationElements, _data, _mask);

                _mm256_store_pd(&arr[index], _lastIterationElements);
            }
            else if (lastIterationCount == 4 && index != i)
            {

                _mm256_store_pd(&arr[i], _data);
                index += 4;
                _lastIterationElements = _mm256_load_pd(&arr[index]);
                _lastIterationMask = _mm256_cmp_pd(_lastIterationElements, _pivot, _CMP_LT_OQ);
                lastIterationMask = _mm256_movemask_pd(_lastIterationMask);

                _indicesToPermute = lookUpTableLeftPermute[lastIterationMask];
                _lastIterationElements = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_lastIterationElements), _indicesToPermute));
                _lastIterationMask = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_lastIterationMask), _indicesToPermute));
                lastIterationMask = _mm256_movemask_pd(_lastIterationMask);
                lastIterationCount = _mm_popcnt_u32(lastIterationMask);
                continue;
            }
            _mm256_store_pd(&arr[i], _data);
        }
    }

    size_t j = index + lastIterationCount;

    for (size_t i = finalPos; i < r; i++)
    {
        if (arr[i] < arr[r])
        {

            std::swap(arr[i], arr[j]);
            j++;
        }
    }

    std::swap(arr[j], arr[r]);

    return j;
}

void quickSortSimd(double* arr, int l, int r)
{
    if (l < r)
    {
        size_t pivot = partitionSimd(arr, l, r);

        quickSortSimd(arr, l, pivot - 1);
        quickSortSimd(arr, pivot + 1, r);
    }
}

int main()
{
    const size_t size = 20;
    double arr[size];

    srand(time(nullptr));

    for (size_t i = 0; i < size; i++) arr[i] = static_cast<double>(rand() % 100);

    for (size_t i = 0; i < sizeof(arr) / sizeof(double); i++) std::cout << arr[i] << std::endl;

    //size_t pivot = partitionSimd(arr, 0, (sizeof(arr) / sizeof(double)) - 1);

    quickSortSimd(arr, 0, (sizeof(arr) / sizeof(double)) - 1);

    std::cout << std::endl << std::endl;

    for (size_t i = 0; i < sizeof(arr) / sizeof(double); i++) std::cout << arr[i] << std::endl;

    std::cout << std::endl;

    //std::cout << pivot << std::endl;
}
