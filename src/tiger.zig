// TigerStyle core primitives, always-on release assertions, monotonic timing,
// fixed-capacity ByteWriter, and minimal POSIX/libc C-ABI declarations.
// Zero dynamic allocations during request or token execution.

const std = @import("std");
const builtin = @import("builtin");

pub const is_windows: bool = (builtin.os.tag == .windows);

comptime {
    if (builtin.cpu.arch != .x86_64 or (builtin.os.tag != .linux and builtin.os.tag != .windows)) {
        @compileError("squid4070 supports linux x86_64 and windows x86_64");
    }
}

/// TigerStyle always-on assertion: remains active even in ReleaseFast builds.
/// Crashes loudly on invariant violation rather than silently corrupting state.
pub inline fn assert(condition: bool) void {
    if (!condition) {
        @branchHint(.cold);
        @panic("TigerStyle assertion failed");
    }
}

/// Always-on assertion for CUDA runtime status codes (`0 == cudaSuccess`).
pub inline fn assert_cuda(error_code: i32) void {
    if (error_code != 0) {
        @branchHint(.cold);
        log_error_i32("CUDA runtime call failed with error_code=", error_code);
        @panic("CUDA assertion failed");
    }
}

/// Align `value` upward to a power-of-two `alignment` boundary.
pub inline fn align_up_u64(value: u64, alignment: u64) u64 {
    assert(alignment > 0);
    assert((alignment & (alignment - 1)) == 0);
    const mask: u64 = alignment - 1;
    const result: u64 = (value + mask) & ~mask;
    assert(result >= value);
    assert((result & mask) == 0);
    return result;
}

pub inline fn min_u64(a: u64, b: u64) u64 {
    return if (a < b) a else b;
}

pub inline fn max_u64(a: u64, b: u64) u64 {
    return if (a > b) a else b;
}

pub inline fn min_i32(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}

pub inline fn max_i32(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}

/// FNV-1a 64-bit hash for flat static open-addressed tables.
pub inline fn hash_bytes_u64(bytes: []const u8) u64 {
    var hash: u64 = 14695981039346656037;
    for (bytes) |byte| {
        hash ^= @as(u64, byte);
        hash *%= 1099511628211;
    }
    return if (hash == 0) 1 else hash;
}

/// SplitMix64 mixer for 64-bit integer keys (e.g., BPE token pairs).
pub inline fn hash_pair_u64(key: u64) u64 {
    var mixed: u64 = key +% 0x9e3779b97f4a7c15;
    mixed = (mixed ^ (mixed >> 30)) *% 0xbf58476d1ce4e5b9;
    mixed = (mixed ^ (mixed >> 27)) *% 0x94d049bb133111eb;
    mixed = mixed ^ (mixed >> 31);
    return if (mixed == 0) 1 else mixed;
}

// --- POSIX / Libc C-ABI Declarations ---

pub const Timespec = extern struct {
    tv_sec: i64,
    tv_nsec: i64,
};

pub const Timeval = extern struct {
    tv_sec: i64,
    tv_usec: i64,
};

pub const SockaddrIn = extern struct {
    sin_family: u16,
    sin_port: u16,
    sin_addr: u32,
    sin_zero: [8]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
};

comptime {
    if (@sizeOf(SockaddrIn) != 16) {
        @compileError("SockaddrIn must be exactly 16 bytes");
    }
    if (@sizeOf(Timeval) != 16) {
        @compileError("Timeval must be exactly 16 bytes");
    }
}

pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_CREAT: i32 = if (is_windows) 0x100 else 0x40;
pub const O_TRUNC: i32 = 0x200;
const O_BINARY: i32 = if (is_windows) 0x8000 else 0;
pub const SEEK_SET: i32 = 0;
pub const SEEK_END: i32 = 2;
pub const PROT_READ: i32 = 1;
pub const MAP_PRIVATE: i32 = 2;
pub const MAP_FAILED_ADDR: usize = std.math.maxInt(usize);

pub const CLOCK_REALTIME: i32 = 0;
pub const CLOCK_MONOTONIC: i32 = 1;

pub const AF_INET: i32 = 2;
pub const SOCK_STREAM: i32 = 1;
pub const SOL_SOCKET: i32 = if (is_windows) 0xffff else 1;
pub const SO_REUSEADDR: i32 = if (is_windows) 4 else 2;
pub const SO_RCVTIMEO: i32 = if (is_windows) 0x1006 else 20;
pub const SO_SNDTIMEO: i32 = if (is_windows) 0x1005 else 21;
pub const IPPROTO_TCP: i32 = 6;
pub const TCP_NODELAY: i32 = 1;
pub const MSG_NOSIGNAL: i32 = if (is_windows) 0 else 0x4000;
pub const EINTR: i32 = 4;

pub const SIGINT: i32 = 2;
pub const SIGPIPE: i32 = 13;
pub const SIGTERM: i32 = 15;

pub const SignalHandler = ?*const fn (i32) callconv(.c) void;
pub const SIG_IGN: SignalHandler = @ptrFromInt(1);

pub const SocketFd = if (is_windows) usize else i32;
pub const INVALID_SOCKET: SocketFd = if (is_windows) std.math.maxInt(usize) else -1;

pub extern "c" fn malloc(size: usize) ?*anyopaque;
pub extern "c" fn calloc(count: usize, size: usize) ?*anyopaque;
pub extern "c" fn free(ptr: ?*anyopaque) void;
pub extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

const sys = if (is_windows) struct {
    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x0004;
    const WSAEINTR: i32 = 10004;
    const S_IREAD_IWRITE: i32 = 0o600;

    const FILETIME = extern struct {
        dwLowDateTime: u32,
        dwHighDateTime: u32,
    };

    const WSAData = extern struct {
        raw: [512]u8 align(8) = @splat(0),
    };

    const CtrlHandler = *const fn (ctrl_type: u32) callconv(.winapi) i32;

    extern "c" fn _open(pathname: [*:0]const u8, flags: i32, ...) i32;
    extern "c" fn _close(fd: i32) i32;
    extern "c" fn _read(fd: i32, buffer: [*]u8, count: u32) i32;
    extern "c" fn _write(fd: i32, buffer: [*]const u8, count: u32) i32;
    extern "c" fn _lseeki64(fd: i32, offset: i64, whence: i32) i64;
    extern "c" fn _get_osfhandle(fd: i32) isize;
    extern "c" fn _errno() *i32;

    extern "kernel32" fn CreateFileMappingA(
        hFile: ?*anyopaque,
        lpAttributes: ?*anyopaque,
        flProtect: u32,
        dwMaximumSizeHigh: u32,
        dwMaximumSizeLow: u32,
        lpName: ?[*:0]const u8,
    ) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn MapViewOfFile(
        hFileMappingObject: ?*anyopaque,
        dwDesiredAccess: u32,
        dwFileOffsetHigh: u32,
        dwFileOffsetLow: u32,
        dwNumberOfBytesToMap: usize,
    ) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn UnmapViewOfFile(lpBaseAddress: *const anyopaque) callconv(.winapi) i32;
    extern "kernel32" fn CloseHandle(hObject: ?*anyopaque) callconv(.winapi) i32;
    extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.winapi) i32;
    extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;
    extern "kernel32" fn GetSystemTimeAsFileTime(lpSystemTimeAsFileTime: *FILETIME) callconv(.winapi) void;
    extern "kernel32" fn SetConsoleCtrlHandler(HandlerRoutine: ?CtrlHandler, Add: i32) callconv(.winapi) i32;

    extern "ws2_32" fn WSAStartup(wVersionRequested: u16, lpWSAData: *WSAData) callconv(.winapi) i32;
    extern "ws2_32" fn WSAGetLastError() callconv(.winapi) i32;
    extern "ws2_32" fn socket(af: i32, sock_type: i32, protocol: i32) callconv(.winapi) usize;
    extern "ws2_32" fn closesocket(s: usize) callconv(.winapi) i32;
    extern "ws2_32" fn setsockopt(s: usize, level: i32, optname: i32, optval: *const anyopaque, optlen: i32) callconv(.winapi) i32;
    extern "ws2_32" fn bind(s: usize, name: *const SockaddrIn, namelen: i32) callconv(.winapi) i32;
    extern "ws2_32" fn listen(s: usize, backlog: i32) callconv(.winapi) i32;
    extern "ws2_32" fn accept(s: usize, addr: ?*SockaddrIn, addrlen: ?*i32) callconv(.winapi) usize;
    extern "ws2_32" fn recv(s: usize, buf: [*]u8, len: i32, flags: i32) callconv(.winapi) i32;
    extern "ws2_32" fn send(s: usize, buf: [*]const u8, len: i32, flags: i32) callconv(.winapi) i32;
    extern "ws2_32" fn inet_pton(af: i32, src: [*:0]const u8, dst: *u32) callconv(.winapi) i32;
} else struct {
    extern "c" fn open(pathname: [*:0]const u8, flags: i32, ...) i32;
    extern "c" fn close(fd: i32) i32;
    extern "c" fn read(fd: i32, buffer: [*]u8, count: usize) isize;
    extern "c" fn write(fd: i32, buffer: [*]const u8, count: usize) isize;
    extern "c" fn lseek(fd: i32, offset: i64, whence: i32) i64;
    extern "c" fn mmap(addr: ?*anyopaque, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) *anyopaque;
    extern "c" fn munmap(addr: *anyopaque, length: usize) i32;
    extern "c" fn clock_gettime(clock_id: i32, tp: *Timespec) i32;
    extern "c" fn socket(domain: i32, sock_type: i32, protocol: i32) i32;
    extern "c" fn setsockopt(fd: i32, level: i32, optname: i32, optval: *const anyopaque, optlen: u32) i32;
    extern "c" fn bind(fd: i32, addr: *const SockaddrIn, addrlen: u32) i32;
    extern "c" fn listen(fd: i32, backlog: i32) i32;
    extern "c" fn accept(fd: i32, addr: ?*SockaddrIn, addrlen: ?*u32) i32;
    extern "c" fn recv(fd: i32, buffer: [*]u8, length: usize, flags: i32) isize;
    extern "c" fn send(fd: i32, buffer: [*]const u8, length: usize, flags: i32) isize;
    extern "c" fn inet_pton(af: i32, src: [*:0]const u8, dst: *u32) i32;
    extern "c" fn signal(signum: i32, handler: SignalHandler) SignalHandler;
    extern "c" fn __errno_location() *i32;
};

pub fn open(pathname: [*:0]const u8, flags: i32) i32 {
    if (is_windows) return sys._open(pathname, flags | O_BINARY);
    return sys.open(pathname, flags);
}

pub fn open_create(pathname: [*:0]const u8, flags: i32, mode: u32) i32 {
    if (is_windows) {
        return sys._open(pathname, flags | O_BINARY, sys.S_IREAD_IWRITE);
    }
    return sys.open(pathname, flags, mode);
}

pub inline fn close(fd: i32) i32 {
    if (is_windows) return sys._close(fd);
    return sys.close(fd);
}

pub fn read(fd: i32, buffer: [*]u8, count: usize) isize {
    if (is_windows) {
        const chunk: u32 = @intCast(@min(count, 0x4000_0000));
        return @as(isize, sys._read(fd, buffer, chunk));
    }
    return sys.read(fd, buffer, count);
}

pub fn write(fd: i32, buffer: [*]const u8, count: usize) isize {
    if (is_windows) {
        const chunk: u32 = @intCast(@min(count, 0x4000_0000));
        return @as(isize, sys._write(fd, buffer, chunk));
    }
    return sys.write(fd, buffer, count);
}

pub inline fn lseek(fd: i32, offset: i64, whence: i32) i64 {
    if (is_windows) return sys._lseeki64(fd, offset, whence);
    return sys.lseek(fd, offset, whence);
}

pub fn mmap(addr: ?*anyopaque, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) *anyopaque {
    if (is_windows) {
        assert(addr == null);
        assert(prot == PROT_READ and flags == MAP_PRIVATE);
        assert(length > 0 and offset >= 0);
        const os_handle_int = sys._get_osfhandle(fd);
        if (os_handle_int == -1) return @ptrFromInt(MAP_FAILED_ADDR);
        const os_handle: ?*anyopaque = @ptrFromInt(@as(usize, @bitCast(os_handle_int)));
        const mapping = sys.CreateFileMappingA(os_handle, null, sys.PAGE_READONLY, 0, 0, null) orelse return @ptrFromInt(MAP_FAILED_ADDR);
        defer _ = sys.CloseHandle(mapping);
        const off_u64: u64 = @intCast(offset);
        const view = sys.MapViewOfFile(
            mapping,
            sys.FILE_MAP_READ,
            @intCast(off_u64 >> 32),
            @intCast(off_u64 & 0xffff_ffff),
            length,
        ) orelse return @ptrFromInt(MAP_FAILED_ADDR);
        return view;
    }
    return sys.mmap(addr, length, prot, flags, fd, offset);
}

pub fn munmap(addr: *anyopaque, length: usize) i32 {
    if (is_windows) {
        return if (sys.UnmapViewOfFile(addr) != 0) 0 else -1;
    }
    return sys.munmap(addr, length);
}

pub inline fn htons(value: u16) u16 {
    return @byteSwap(value);
}

pub inline fn get_errno() i32 {
    if (is_windows) return sys._errno().*;
    return sys.__errno_location().*;
}

const StopFn = *const fn () void;
var g_stop_fn: ?StopFn = null;

fn posix_stop_handler(_: i32) callconv(.c) void {
    if (g_stop_fn) |cb| cb();
}

fn win_ctrl_handler(ctrl_type: u32) callconv(.winapi) i32 {
    if (ctrl_type == 0 or ctrl_type == 1 or ctrl_type == 2) {
        if (g_stop_fn) |cb| cb();
        return 1;
    }
    return 0;
}

pub fn install_stop_handlers(cb: StopFn) void {
    g_stop_fn = cb;
    if (is_windows) {
        _ = sys.SetConsoleCtrlHandler(win_ctrl_handler, 1);
    } else {
        _ = sys.signal(SIGPIPE, SIG_IGN);
        _ = sys.signal(SIGINT, posix_stop_handler);
        _ = sys.signal(SIGTERM, posix_stop_handler);
    }
}

pub inline fn is_valid_socket(sock: SocketFd) bool {
    if (is_windows) return sock != INVALID_SOCKET;
    return sock >= 0;
}

pub fn net_init() bool {
    if (is_windows) {
        var wsa_data: sys.WSAData = .{};
        return sys.WSAStartup(0x0202, &wsa_data) == 0;
    }
    return true;
}

pub inline fn net_socket_tcp() SocketFd {
    return sys.socket(AF_INET, SOCK_STREAM, 0);
}

pub inline fn net_close(sock: SocketFd) void {
    if (is_windows) {
        _ = sys.closesocket(sock);
    } else {
        _ = sys.close(sock);
    }
}

pub inline fn net_set_reuseaddr(sock: SocketFd) void {
    var reuse: i32 = 1;
    _ = sys.setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, @ptrCast(&reuse), @sizeOf(i32));
}

pub inline fn net_set_nodelay(sock: SocketFd) void {
    var nodelay: i32 = 1;
    _ = sys.setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, @ptrCast(&nodelay), @sizeOf(i32));
}

pub fn net_set_timeouts_ms(sock: SocketFd, timeout_ms: u32) void {
    if (is_windows) {
        var ms: u32 = timeout_ms;
        _ = sys.setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, @ptrCast(&ms), @sizeOf(u32));
        _ = sys.setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, @ptrCast(&ms), @sizeOf(u32));
    } else {
        const tv: Timeval = .{
            .tv_sec = @intCast(timeout_ms / 1000),
            .tv_usec = @intCast((timeout_ms % 1000) * 1000),
        };
        _ = sys.setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, @ptrCast(&tv), @sizeOf(Timeval));
        _ = sys.setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, @ptrCast(&tv), @sizeOf(Timeval));
    }
}

pub inline fn net_inet_pton_ipv4(ip_z: [*:0]const u8, dst: *u32) i32 {
    return sys.inet_pton(AF_INET, ip_z, dst);
}

pub inline fn net_bind_ipv4(sock: SocketFd, addr: *const SockaddrIn) bool {
    return sys.bind(sock, addr, @sizeOf(SockaddrIn)) == 0;
}

pub inline fn net_listen(sock: SocketFd, backlog: i32) bool {
    return sys.listen(sock, backlog) == 0;
}

pub inline fn net_accept(sock: SocketFd) SocketFd {
    return sys.accept(sock, null, null);
}

pub inline fn net_recv(sock: SocketFd, buffer: [*]u8, length: usize) isize {
    if (is_windows) {
        return @as(isize, sys.recv(sock, buffer, @intCast(@min(length, 0x4000_0000)), 0));
    }
    return sys.recv(sock, buffer, length, 0);
}

pub inline fn net_send(sock: SocketFd, buffer: [*]const u8, length: usize) isize {
    if (is_windows) {
        return @as(isize, sys.send(sock, buffer, @intCast(@min(length, 0x4000_0000)), 0));
    }
    return sys.send(sock, buffer, length, MSG_NOSIGNAL);
}

pub inline fn net_last_error_is_eintr() bool {
    if (is_windows) return sys.WSAGetLastError() == sys.WSAEINTR;
    return get_errno() == EINTR;
}

/// Monotonic wall clock in nanoseconds.
pub fn clock_monotonic_ns() u64 {
    if (is_windows) {
        var ctr: i64 = 0;
        var freq: i64 = 0;
        assert(sys.QueryPerformanceCounter(&ctr) != 0);
        assert(sys.QueryPerformanceFrequency(&freq) != 0);
        assert(ctr >= 0 and freq > 0);
        return @intCast((@as(u128, @intCast(ctr)) * 1_000_000_000) / @as(u128, @intCast(freq)));
    }
    var ts: Timespec = .{ .tv_sec = 0, .tv_nsec = 0 };
    const rc: i32 = sys.clock_gettime(CLOCK_MONOTONIC, &ts);
    assert(rc == 0);
    assert(ts.tv_sec >= 0);
    assert(ts.tv_nsec >= 0 and ts.tv_nsec < 1_000_000_000);
    return @as(u64, @intCast(ts.tv_sec)) * 1_000_000_000 + @as(u64, @intCast(ts.tv_nsec));
}

/// UNIX epoch timestamp in seconds (for OpenAI JSON `created` fields).
pub fn epoch_seconds() i64 {
    if (is_windows) {
        var ft: sys.FILETIME = undefined;
        sys.GetSystemTimeAsFileTime(&ft);
        const ticks: u64 = (@as(u64, ft.dwHighDateTime) << 32) | @as(u64, ft.dwLowDateTime);
        const unix_offset: u64 = 116_444_736_000_000_000;
        if (ticks < unix_offset) return 0;
        return @intCast((ticks - unix_offset) / 10_000_000);
    }
    var ts: Timespec = .{ .tv_sec = 0, .tv_nsec = 0 };
    const rc: i32 = sys.clock_gettime(CLOCK_REALTIME, &ts);
    assert(rc == 0);
    assert(ts.tv_sec >= 0);
    return ts.tv_sec;
}

/// Allocate zero-initialized host memory at startup (never called on hot paths).
pub fn alloc_host_zeroed(comptime T: type, count: usize) []T {
    assert(count > 0);
    const raw: ?*anyopaque = calloc(count, @sizeOf(T));
    assert(raw != null);
    const typed: [*]T = @ptrCast(@alignCast(raw.?));
    return typed[0..count];
}

/// Free startup-allocated host memory during teardown.
pub fn free_host_slice(comptime T: type, slice: []T) void {
    assert(slice.len > 0);
    free(@ptrCast(slice.ptr));
}

// --- Zero-Allocation Fixed-Buffer Writer & Direct POSIX Stdout/Stderr ---

pub const ByteWriter = struct {
    buffer: []u8,
    length: usize,

    pub fn init(buffer: []u8) ByteWriter {
        assert(buffer.len > 0);
        return .{ .buffer = buffer, .length = 0 };
    }

    pub fn reset(self: *ByteWriter) void {
        assert(self.length <= self.buffer.len);
        self.length = 0;
    }

    pub fn slice(self: *const ByteWriter) []const u8 {
        assert(self.length <= self.buffer.len);
        return self.buffer[0..self.length];
    }

    pub fn remaining(self: *const ByteWriter) usize {
        assert(self.length <= self.buffer.len);
        return self.buffer.len - self.length;
    }

    pub fn append_slice(self: *ByteWriter, text: []const u8) void {
        assert(self.length <= self.buffer.len);
        assert(text.len <= self.buffer.len - self.length);
        @memcpy(self.buffer[self.length .. self.length + text.len], text);
        self.length += text.len;
    }

    pub fn try_append_slice(self: *ByteWriter, text: []const u8) bool {
        assert(self.length <= self.buffer.len);
        if (text.len > self.buffer.len - self.length) return false;
        @memcpy(self.buffer[self.length .. self.length + text.len], text);
        self.length += text.len;
        return true;
    }

    pub fn append_byte(self: *ByteWriter, byte: u8) void {
        assert(self.length < self.buffer.len);
        self.buffer[self.length] = byte;
        self.length += 1;
    }

    pub fn try_append_byte(self: *ByteWriter, byte: u8) bool {
        assert(self.length <= self.buffer.len);
        if (self.length >= self.buffer.len) return false;
        self.buffer[self.length] = byte;
        self.length += 1;
        return true;
    }

    pub fn append_fmt(self: *ByteWriter, comptime fmt: []const u8, args: anytype) void {
        assert(self.length <= self.buffer.len);
        const written = std.fmt.bufPrint(self.buffer[self.length..], fmt, args) catch {
            @panic("ByteWriter.append_fmt buffer overflow");
        };
        self.length += written.len;
        assert(self.length <= self.buffer.len);
    }

    pub fn try_append_fmt(self: *ByteWriter, comptime fmt: []const u8, args: anytype) bool {
        assert(self.length <= self.buffer.len);
        const written = std.fmt.bufPrint(self.buffer[self.length..], fmt, args) catch return false;
        self.length += written.len;
        return true;
    }

    pub fn try_append_json_escaped(self: *ByteWriter, text: []const u8) bool {
        assert(self.length <= self.buffer.len);
        const saved = self.length;
        if (!self.try_append_byte('"')) return false;
        const hex_digits = "0123456789abcdef";
        for (text) |byte| {
            const ok: bool = switch (byte) {
                '"' => self.try_append_slice("\\\""),
                '\\' => self.try_append_slice("\\\\"),
                '\n' => self.try_append_slice("\\n"),
                '\r' => self.try_append_slice("\\r"),
                '\t' => self.try_append_slice("\\t"),
                0x08 => self.try_append_slice("\\b"),
                0x0c => self.try_append_slice("\\f"),
                else => blk: {
                    if (byte < 0x20) {
                        var esc: [6]u8 = .{ '\\', 'u', '0', '0', hex_digits[(byte >> 4) & 0x0f], hex_digits[byte & 0x0f] };
                        break :blk self.try_append_slice(&esc);
                    } else {
                        break :blk self.try_append_byte(byte);
                    }
                },
            };
            if (!ok) {
                self.length = saved;
                return false;
            }
        }
        if (!self.try_append_byte('"')) {
            self.length = saved;
            return false;
        }
        return true;
    }

    pub fn append_json_escaped(self: *ByteWriter, text: []const u8) void {
        assert(self.try_append_json_escaped(text));
    }

    pub fn append_c_string_terminated(self: *ByteWriter, text: []const u8) [*:0]const u8 {
        const start = self.length;
        self.append_slice(text);
        self.append_byte(0);
        return @ptrCast(self.buffer[start..self.length].ptr);
    }
};

pub fn write_fd_all_checked(fd: i32, bytes: []const u8) bool {
    assert(fd >= 0);
    var sent: usize = 0;
    const iterations_max: usize = bytes.len + 64;
    var iteration: usize = 0;
    while (sent < bytes.len and iteration < iterations_max) : (iteration += 1) {
        const written: isize = write(fd, bytes[sent..].ptr, bytes.len - sent);
        if (written <= 0) {
            if (written < 0 and get_errno() == EINTR) continue;
            return false;
        }
        sent += @intCast(written);
    }
    return sent == bytes.len;
}

pub fn write_fd_all(fd: i32, bytes: []const u8) void {
    _ = write_fd_all_checked(fd, bytes);
}

pub fn print_stdout(comptime fmt: []const u8, args: anytype) void {
    var stack_buffer: [8192]u8 = @splat(0);
    const text = std.fmt.bufPrint(&stack_buffer, fmt, args) catch blk: {
        const len = std.mem.indexOfScalar(u8, &stack_buffer, 0) orelse stack_buffer.len;
        break :blk stack_buffer[0..len];
    };
    write_fd_all(1, text);
}

pub fn print_stderr(comptime fmt: []const u8, args: anytype) void {
    var stack_buffer: [4096]u8 = @splat(0);
    const text = std.fmt.bufPrint(&stack_buffer, fmt, args) catch blk: {
        const len = std.mem.indexOfScalar(u8, &stack_buffer, 0) orelse stack_buffer.len;
        break :blk stack_buffer[0..len];
    };
    write_fd_all(2, text);
}

fn log_error_i32(prefix: []const u8, value: i32) void {
    print_stderr("{s}{d}\n", .{ prefix, value });
}

/// Build a null-terminated path `<directory>/<suffix>` inside caller stack buffer.
pub fn format_path_z(buffer: []u8, directory: []const u8, suffix: []const u8) [*:0]const u8 {
    assert(buffer.len > directory.len + suffix.len + 1);
    assert(directory.len > 0);
    @memcpy(buffer[0..directory.len], directory);
    @memcpy(buffer[directory.len .. directory.len + suffix.len], suffix);
    buffer[directory.len + suffix.len] = 0;
    return @ptrCast(buffer[0 .. directory.len + suffix.len + 1].ptr);
}
