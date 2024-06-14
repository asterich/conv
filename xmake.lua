add_rules("mode.debug", "mode.release")

-- define toolchain
toolchain("intel-oneapi")

    -- set toolset
    set_toolset("cc", "icx")
    set_toolset("cxx", "icpx", "icx")
    set_toolset("ld", "icpx", "icx")
    set_toolset("sh", "icpx", "icx")
    set_toolset("ar", "ar")
    set_toolset("strip", "strip")
    set_toolset("as", "icx")

    add_defines("INTEL_ONEAPI")

    -- check toolchain
    on_check(function (toolchain)
        return import("lib.detect.find_tool")("icx")
    end)

    -- on load
    on_load(function (toolchain)

        -- add march flags
        local march
        if toolchain:is_arch("x86_64", "x64") then
            march = "-m64"
        elseif toolchain:is_arch("i386", "x86") then
            march = "-m32"
        end
        if march then
            toolchain:add("cxflags", march)
            toolchain:add("asflags", march)
            toolchain:add("ldflags", march)
            toolchain:add("shflags", march)
        end

        -- get icx environments
        local icxenv = toolchain:config("icxenv")
        if icxenv then
            local ldname = is_host("macosx") and "DYLD_LIBRARY_PATH" or "LD_LIBRARY_PATH"
            toolchain:add("runenvs", ldname, icxenv.libdir)
        end
    end)
toolchain_end()

add_requires("onednn", {configs = {cpu_runtime = "omp"}})

target("conv")
    set_kind("binary")
    add_files("src/*.cpp")
    add_includedirs("include")
    add_includedirs("$(env VTUNE_PROFILER_DIR)/sdk/include")
    add_includedirs("$(env MKLROOT)/include")
    add_packages("onednn")
    set_rundir("$(projectdir)")
    add_cxxflags("-std=c++20 -O3 -g -march=native")
    if is_config("toolchain", "intel-oneapi") then
        add_cxxflags("-xCORE-AVX512 -qopenmp -qopt-report -vec-threshold0", {force = true})
        add_ldflags("-liomp5 -g -Rno-debug-disables-optimization")
    else
        add_cxxflags("-fopenmp -mavx512f")
        add_ldflags("-fopenmp")
    end
    add_ldflags("-L$(env VTUNE_PROFILER_DIR)/sdk/lib64 -littnotify")
    add_ldflags("-Wl,--start-group $(env MKLROOT)/lib/libmkl_intel_ilp64.a $(env MKLROOT)/lib/libmkl_intel_thread.a $(env MKLROOT)/lib/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl", {force = true})

