#!/bin/bash

# Скрипт для адаптации nvdiffrast под ROCm 7.1 (архитектура gfx1201 / Wave64)
# Запускать из корня репозитория nvdiffrast

echo "--- Начало процесса патчинга для ROCm 7.1 (Wave64) ---"

# 0. Установка системных зависимостей для линковки (на всякий случай)
echo "[0/5] Проверка системных библиотек..."
sudo apt install -y hipsparse-dev hipblas-dev rocthrust-dev hipcub-dev 2>/dev/null

# 1. Исправление фундаментальных заголовков
echo "[1/5] Настройка framework.h и инклудов..."
sed -i '2i #ifndef USE_ROCM\n#define USE_ROCM\n#endif' csrc/common/framework.h
sed -i '3i #ifndef __HIP_PLATFORM_AMD__\n#define __HIP_PLATFORM_AMD__ 1\n#endif' csrc/common/framework.h
# Добавляем инклуд расширения во все файлы, если его там нет
find csrc/ -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.hip" \) -exec sed -i '1i #include <torch/extension.h>' {} + 2>/dev/null

# 2. Массовая замена масок и типов для Wave64 (защита \b предотвращает ullll)
echo "[2/5] Исправление масок синхронизации для архитектуры Wave64..."
find csrc/ -type f \( -name "*.cu" -o -name "*.hip" -o -name "*.inl" -o -name "*.h" \) -exec sed -i 's/0xffffffffu\b/0xffffffffffffffffull/g' {} +
find csrc/ -type f \( -name "*.cu" -o -name "*.hip" -o -name "*.inl" -o -name "*.h" \) -exec sed -i 's/~0u\b/~0ull/g' {} +

# Исправляем типы переменных масок в коде растризатора
find csrc/common/hipraster/impl/ -type f -name "*.inl" -exec sed -i -E 's/\bU32\b ([a-zA-Z0-9_]*Mask)/U64 \1/g' {} +
sed -i 's/U32 getLaneMask/U64 getLaneMask/g' csrc/common/hipraster/impl/Util.inl
# Исправляем amask в antialias (критично для Wave64)
find csrc/ -type f -name "antialias.*" -exec sed -i 's/unsigned int amask/unsigned long long amask/g' {} +

# 3. Удаление NVIDIA-специфичного ассемблера из Util.inl
echo "[3/5] Замена PTX-ассемблера на стандартный C++..."
UTIL_PATH="csrc/common/hipraster/impl/Util.inl"
if [ -f "$UTIL_PATH" ]; then
    sed -i 's/asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(v) : "f"(a));/v = (S32)roundf(a);/g' $UTIL_PATH
    sed -i 's/asm("cvt.rni.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a));/v = (U32)roundf(max(0.0f, a));/g' $UTIL_PATH
    sed -i 's/asm("cvt.rmi.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a));/v = (U32)floorf(max(0.0f, a));/g' $UTIL_PATH
    sed -i 's/asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(v) : "f"(a));/v = (U32)roundf(max(0.0f, min(255.0f, a)));/g' $UTIL_PATH
    sed -i 's/asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));/v = min(min(a, b), c);/g' $UTIL_PATH
    sed -i 's/asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));/v = max(max(a, b), c);/g' $UTIL_PATH
    sed -i 's/asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));/v = a + b + c;/g' $UTIL_PATH
    sed -i 's/asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(c), "r"(b));/v = a - b + c;/g' $UTIL_PATH
    sed -i 's/asm("vadd.u32.s32.s32.sat.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));/v = max(0, min(a + b, c));/g' $UTIL_PATH
    sed -i 's/asm("slct.s32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));/v = (c >= 0) ? a : b;/g' $UTIL_PATH
    sed -i 's/asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v));/r = (v == 0) ? -1 : 31 - __clz(v);/g' $UTIL_PATH
    sed -i 's/asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));/v = 0;/g' $UTIL_PATH
fi

# 4. Исправление ошибок синтаксиса в PyTorch обертках
echo "[4/5] Исправление narrowing conversion и пространств имен..."
# Narrowing фикс (используем static_cast)
find csrc/torch/ -name "torch_antialias*" -exec sed -i 's/{(uint64_t)p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles) * 4}/{static_cast<long>((uint64_t)p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles) * 4)}/g' {} +

# Исправление математики текстур
find csrc/ -name "texture_kernel.*" -exec sed -i 's/__frcp_rz/__frcp_rn/g' {} +

# Пространства имен и заголовки
find csrc/torch/ -type f -exec sed -i 's/at::hip::/at::cuda::/g' {} +
find csrc/torch/ -type f -exec sed -i 's/c10::hip::/at::cuda::/g' {} +
find csrc/torch/ -type f -exec sed -i 's/c10\/cuda\/CUDAGuard.h/c10\/hip\/HIPGuard.h/g' {} +
find csrc/ -type f -exec sed -i 's/cudaGetLastError/hipGetLastError/g' {} +

# 5. Маскировка CUDA библиотек под ROCm
echo "[5/5] Создание системных ссылок для компилятора..."
sudo ln -s /opt/rocm/include/hipsparse/hipsparse.h /opt/rocm/include/cusparse.h 2>/dev/null
sudo ln -s /opt/rocm/include/hipblas/hipblas.h /opt/rocm/include/cublas_v2.h 2>/dev/null
sudo ln -s /opt/rocm/include/hip/hip_runtime_api.h /opt/rocm/include/cuda_runtime_api.h 2>/dev/null

echo "--- Патчинг завершен! ---"
echo "Теперь выполните:"
echo "rm -rf build/"
echo "export PYTORCH_ROCM_ARCH=gfx1201"
echo "python setup.py install"
