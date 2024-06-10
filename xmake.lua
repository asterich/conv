add_rules("mode.debug", "mode.release")

add_requires("openmp", {system = true})

target("conv")
    set_kind("binary")
    add_files("src/*.cpp")
    add_includedirs("include")
