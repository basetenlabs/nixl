/*
 * NIXL POSIX Plugin VRAM Bounce Buffer Tests
 *
 * Tests FILE_SEG <-> VRAM_SEG transfers via pinned DRAM bounce buffer,
 * including swapped local/remote orientation handling.
 *
 * Requires: CUDA GPU, CUFILE_ENV_PATH_JSON with compat mode config
 */
#include <filesystem>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <cassert>
#include <cstring>
#include <string>
#include <cuda_runtime.h>
#include "nixl.h"
#include "nixl_descriptors.h"

namespace {

constexpr size_t SMALL_SIZE  = 4096;
constexpr size_t MEDIUM_SIZE = 512 * 1024;   // 512KB
constexpr int    NUM_BLOCKS  = 16;
constexpr char   TEST_DIR[]  = "/tmp/nixl_posix_vram_test";

#define CHECK_IO(expr, expected) do { \
    ssize_t _r = (expr); \
    if (_r != (ssize_t)(expected)) { \
        std::cerr << "IO error at " << __LINE__ << ": got " << _r << std::endl; \
        abort(); \
    } \
} while(0)

struct TestFile {
    int fd;
    std::string path;

    TestFile(const std::string& name, size_t size, char fill) {
        path = std::string(TEST_DIR) + "/" + name;
        fd = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
        assert(fd >= 0);
        std::vector<char> buf(size, fill);
        CHECK_IO(pwrite(fd, buf.data(), size, 0), size);
        fsync(fd);
    }
    ~TestFile() {
        if (fd >= 0) close(fd);
        unlink(path.c_str());
    }
};

struct GpuBuf {
    void* ptr;
    size_t size;

    GpuBuf(size_t sz) : ptr(nullptr), size(sz) {
        cudaError_t ret __attribute__((unused)) = cudaMalloc(&ptr, sz);
        assert(ret == cudaSuccess);
        cudaMemset(ptr, 0, sz);
    }
    ~GpuBuf() { if (ptr) cudaFree(ptr); }

    void fill(char c) {
        std::vector<char> buf(size, c);
        cudaMemcpy(ptr, buf.data(), size, cudaMemcpyHostToDevice);
    }
    bool verify(char expected) {
        std::vector<char> buf(size);
        cudaMemcpy(buf.data(), ptr, size, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < size; i++) {
            if (buf[i] != expected) return false;
        }
        return true;
    }
    bool verify_block(size_t offset, size_t len, char expected) {
        std::vector<char> buf(len);
        cudaMemcpy(buf.data(), (char*)ptr + offset, len, cudaMemcpyDeviceToHost);
        return buf[0] == expected && buf[len - 1] == expected;
    }
};

nixl_status_t wait_xfer(nixlAgent& agent, nixlXferReqH* handle) {
    for (int i = 0; i < 200; i++) {
        nixl_status_t s = agent.getXferStatus(handle);
        if (s == NIXL_SUCCESS) return NIXL_SUCCESS;
        if (s != NIXL_IN_PROG) return s;
        usleep(50000);
    }
    return NIXL_ERR_UNKNOWN;
}

nixlBackendH* create_posix_backend(nixlAgent& agent) {
    nixl_b_params_t params;
    params["use_posix_aio"] = "true";
    nixlBackendH* backend = nullptr;
    nixl_status_t s = agent.createBackend("POSIX", params, backend);
    if (s != NIXL_SUCCESS) {
        std::cerr << "Failed to create POSIX backend: " << s << std::endl;
        return nullptr;
    }
    return backend;
}

int tests_passed = 0;
int tests_failed = 0;

void report(const char* name, bool passed) {
    if (passed) {
        tests_passed++;
        std::cout << "  PASS: " << name << std::endl;
    } else {
        tests_failed++;
        std::cerr << "  FAIL: " << name << std::endl;
    }
}

// ============================================================
// Test 1: FILE -> VRAM read (normal orientation: local=VRAM, remote=FILE)
// ============================================================
bool test_file_to_vram_read_normal() {
    nixlAgent agent("t1-normal", nixlAgentConfig(true));
    if (!create_posix_backend(agent)) return false;

    TestFile file("t1.dat", SMALL_SIZE, 'A');
    GpuBuf gpu(SMALL_SIZE);

    nixl_reg_dlist_t freg(FILE_SEG);
    freg.addDesc({0, SMALL_SIZE, (uint64_t)file.fd, ""});
    if (agent.registerMem(freg) != NIXL_SUCCESS) return false;

    nixl_reg_dlist_t vreg(VRAM_SEG);
    vreg.addDesc({(uintptr_t)gpu.ptr, SMALL_SIZE, 0, ""});
    if (agent.registerMem(vreg) != NIXL_SUCCESS) return false;

    // Normal: local=VRAM(dst), remote=FILE(src)
    nixl_xfer_dlist_t local_dl(VRAM_SEG), remote_dl(FILE_SEG);
    local_dl.addDesc({(uintptr_t)gpu.ptr, SMALL_SIZE, 0});
    remote_dl.addDesc({0, SMALL_SIZE, (uint64_t)file.fd});

    nixlXferReqH* h = nullptr;
    if (agent.createXferReq(NIXL_READ, local_dl, remote_dl, "t1-normal", h) != NIXL_SUCCESS) return false;
    if (agent.postXferReq(h) < 0) return false;
    if (wait_xfer(agent, h) != NIXL_SUCCESS) return false;

    bool ok = gpu.verify('A');
    agent.releaseXferReq(h);
    return ok;
}

// ============================================================
// Test 2: FILE -> VRAM read (swapped orientation: local=FILE, remote=VRAM)
// This is how dynamo's write_blocks_to passes descriptors for Read ops.
// ============================================================
bool test_file_to_vram_read_swapped() {
    nixlAgent agent("t2-swap", nixlAgentConfig(true));
    if (!create_posix_backend(agent)) return false;

    TestFile file("t2.dat", SMALL_SIZE, 'B');
    GpuBuf gpu(SMALL_SIZE);

    nixl_reg_dlist_t freg(FILE_SEG);
    freg.addDesc({0, SMALL_SIZE, (uint64_t)file.fd, ""});
    if (agent.registerMem(freg) != NIXL_SUCCESS) return false;

    nixl_reg_dlist_t vreg(VRAM_SEG);
    vreg.addDesc({(uintptr_t)gpu.ptr, SMALL_SIZE, 0, ""});
    if (agent.registerMem(vreg) != NIXL_SUCCESS) return false;

    // Swapped: local=FILE(src), remote=VRAM(dst)
    nixl_xfer_dlist_t local_dl(FILE_SEG), remote_dl(VRAM_SEG);
    local_dl.addDesc({0, SMALL_SIZE, (uint64_t)file.fd});
    remote_dl.addDesc({(uintptr_t)gpu.ptr, SMALL_SIZE, 0});

    nixlXferReqH* h = nullptr;
    if (agent.createXferReq(NIXL_READ, local_dl, remote_dl, "t2-swap", h) != NIXL_SUCCESS) return false;
    if (agent.postXferReq(h) < 0) return false;
    if (wait_xfer(agent, h) != NIXL_SUCCESS) return false;

    bool ok = gpu.verify('B');
    agent.releaseXferReq(h);
    return ok;
}

// ============================================================
// Test 3: VRAM -> FILE write (normal orientation: local=VRAM, remote=FILE)
// ============================================================
bool test_vram_to_file_write() {
    nixlAgent agent("t3-write", nixlAgentConfig(true));
    if (!create_posix_backend(agent)) return false;

    TestFile file("t3.dat", SMALL_SIZE, '\0');
    GpuBuf gpu(SMALL_SIZE);
    gpu.fill('C');

    nixl_reg_dlist_t freg(FILE_SEG);
    freg.addDesc({0, SMALL_SIZE, (uint64_t)file.fd, ""});
    if (agent.registerMem(freg) != NIXL_SUCCESS) return false;

    nixl_reg_dlist_t vreg(VRAM_SEG);
    vreg.addDesc({(uintptr_t)gpu.ptr, SMALL_SIZE, 0, ""});
    if (agent.registerMem(vreg) != NIXL_SUCCESS) return false;

    nixl_xfer_dlist_t local_dl(VRAM_SEG), remote_dl(FILE_SEG);
    local_dl.addDesc({(uintptr_t)gpu.ptr, SMALL_SIZE, 0});
    remote_dl.addDesc({0, SMALL_SIZE, (uint64_t)file.fd});

    nixlXferReqH* h = nullptr;
    if (agent.createXferReq(NIXL_WRITE, local_dl, remote_dl, "t3-write", h) != NIXL_SUCCESS) return false;
    if (agent.postXferReq(h) < 0) return false;
    if (wait_xfer(agent, h) != NIXL_SUCCESS) return false;

    // Read file back and verify
    char rbuf[SMALL_SIZE];
    CHECK_IO(pread(file.fd, rbuf, SMALL_SIZE, 0), SMALL_SIZE);
    bool ok = (rbuf[0] == 'C' && rbuf[SMALL_SIZE - 1] == 'C');
    agent.releaseXferReq(h);
    return ok;
}

// ============================================================
// Test 4: VRAM -> FILE write (swapped orientation: local=FILE, remote=VRAM)
// ============================================================
bool test_vram_to_file_write_swapped() {
    nixlAgent agent("t4-wswap", nixlAgentConfig(true));
    if (!create_posix_backend(agent)) return false;

    TestFile file("t4.dat", SMALL_SIZE, '\0');
    GpuBuf gpu(SMALL_SIZE);
    gpu.fill('D');

    nixl_reg_dlist_t freg(FILE_SEG);
    freg.addDesc({0, SMALL_SIZE, (uint64_t)file.fd, ""});
    if (agent.registerMem(freg) != NIXL_SUCCESS) return false;

    nixl_reg_dlist_t vreg(VRAM_SEG);
    vreg.addDesc({(uintptr_t)gpu.ptr, SMALL_SIZE, 0, ""});
    if (agent.registerMem(vreg) != NIXL_SUCCESS) return false;

    // Swapped write: local=FILE, remote=VRAM
    nixl_xfer_dlist_t local_dl(FILE_SEG), remote_dl(VRAM_SEG);
    local_dl.addDesc({0, SMALL_SIZE, (uint64_t)file.fd});
    remote_dl.addDesc({(uintptr_t)gpu.ptr, SMALL_SIZE, 0});

    nixlXferReqH* h = nullptr;
    if (agent.createXferReq(NIXL_WRITE, local_dl, remote_dl, "t4-wswap", h) != NIXL_SUCCESS) return false;
    if (agent.postXferReq(h) < 0) return false;
    if (wait_xfer(agent, h) != NIXL_SUCCESS) return false;

    char rbuf[SMALL_SIZE];
    CHECK_IO(pread(file.fd, rbuf, SMALL_SIZE, 0), SMALL_SIZE);
    bool ok = (rbuf[0] == 'D' && rbuf[SMALL_SIZE - 1] == 'D');
    agent.releaseXferReq(h);
    return ok;
}

// ============================================================
// Test 5: Large multi-descriptor FILE -> VRAM transfer
// ============================================================
bool test_large_multi_desc() {
    nixlAgent agent("t5-large", nixlAgentConfig(true));
    if (!create_posix_backend(agent)) return false;

    const size_t total = NUM_BLOCKS * MEDIUM_SIZE;
    TestFile file("t5.dat", total, '\0');

    // Write unique pattern per block
    for (int i = 0; i < NUM_BLOCKS; i++) {
        std::vector<char> buf(MEDIUM_SIZE, 'A' + (i % 26));
        CHECK_IO(pwrite(file.fd, buf.data(), MEDIUM_SIZE, i * MEDIUM_SIZE), MEDIUM_SIZE);
    }
    fsync(file.fd);

    GpuBuf gpu(total);

    nixl_reg_dlist_t freg(FILE_SEG);
    nixl_reg_dlist_t vreg(VRAM_SEG);
    for (int i = 0; i < NUM_BLOCKS; i++) {
        freg.addDesc({(uint64_t)(i * MEDIUM_SIZE), MEDIUM_SIZE, (uint64_t)file.fd, ""});
        vreg.addDesc({(uintptr_t)gpu.ptr + i * MEDIUM_SIZE, MEDIUM_SIZE, 0, ""});
    }
    if (agent.registerMem(freg) != NIXL_SUCCESS) return false;
    if (agent.registerMem(vreg) != NIXL_SUCCESS) return false;

    nixl_xfer_dlist_t local_dl(VRAM_SEG), remote_dl(FILE_SEG);
    for (int i = 0; i < NUM_BLOCKS; i++) {
        local_dl.addDesc({(uintptr_t)gpu.ptr + i * MEDIUM_SIZE, MEDIUM_SIZE, 0});
        remote_dl.addDesc({(uint64_t)(i * MEDIUM_SIZE), MEDIUM_SIZE, (uint64_t)file.fd});
    }

    nixlXferReqH* h = nullptr;
    if (agent.createXferReq(NIXL_READ, local_dl, remote_dl, "t5-large", h) != NIXL_SUCCESS) return false;
    if (agent.postXferReq(h) < 0) return false;
    if (wait_xfer(agent, h) != NIXL_SUCCESS) return false;

    bool ok = true;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        if (!gpu.verify_block(i * MEDIUM_SIZE, MEDIUM_SIZE, 'A' + (i % 26))) {
            std::cerr << "Block " << i << " mismatch" << std::endl;
            ok = false;
            break;
        }
    }
    agent.releaseXferReq(h);
    return ok;
}

// ============================================================
// Test 6: DRAM -> FILE still works (regression test)
// ============================================================
bool test_dram_file_regression() {
    nixlAgent agent("t6-dram", nixlAgentConfig(true));
    if (!create_posix_backend(agent)) return false;

    TestFile file("t6.dat", SMALL_SIZE, 'X');

    // Use regular DRAM (posix_memalign)
    void* host_buf = nullptr;
    posix_memalign(&host_buf, 4096, SMALL_SIZE);
    memset(host_buf, 0, SMALL_SIZE);

    nixl_reg_dlist_t freg(FILE_SEG);
    freg.addDesc({0, SMALL_SIZE, (uint64_t)file.fd, ""});
    if (agent.registerMem(freg) != NIXL_SUCCESS) { free(host_buf); return false; }

    nixl_reg_dlist_t dreg(DRAM_SEG);
    dreg.addDesc({(uintptr_t)host_buf, SMALL_SIZE, 0, ""});
    if (agent.registerMem(dreg) != NIXL_SUCCESS) { free(host_buf); return false; }

    // Normal DRAM read: local=DRAM, remote=FILE
    nixl_xfer_dlist_t local_dl(DRAM_SEG), remote_dl(FILE_SEG);
    local_dl.addDesc({(uintptr_t)host_buf, SMALL_SIZE, 0});
    remote_dl.addDesc({0, SMALL_SIZE, (uint64_t)file.fd});

    nixlXferReqH* h = nullptr;
    if (agent.createXferReq(NIXL_READ, local_dl, remote_dl, "t6-dram", h) != NIXL_SUCCESS) { free(host_buf); return false; }
    if (agent.postXferReq(h) < 0) { free(host_buf); return false; }
    if (wait_xfer(agent, h) != NIXL_SUCCESS) { free(host_buf); return false; }

    bool ok = (((char*)host_buf)[0] == 'X' && ((char*)host_buf)[SMALL_SIZE - 1] == 'X');
    agent.releaseXferReq(h);
    free(host_buf);
    return ok;
}

} // namespace

int main() {
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  NIXL POSIX VRAM Bounce Buffer Tests" << std::endl;
    std::cout << "============================================" << std::endl;

    // Create test directory
    std::filesystem::create_directories(TEST_DIR);

    report("FILE->VRAM read (normal)",    test_file_to_vram_read_normal());
    report("FILE->VRAM read (swapped)",   test_file_to_vram_read_swapped());
    report("VRAM->FILE write (normal)",   test_vram_to_file_write());
    report("VRAM->FILE write (swapped)",  test_vram_to_file_write_swapped());
    report("Large multi-desc FILE->VRAM", test_large_multi_desc());
    report("DRAM<->FILE regression",      test_dram_file_regression());

    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  Results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "============================================" << std::endl;

    std::filesystem::remove_all(TEST_DIR);
    return tests_failed > 0 ? 1 : 0;
}
