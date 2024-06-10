add_rules("mode.debug", "mode.release")

add_requires("openmp", {system = true})

target("conv")
    set_kind("binary")
    add_files("src/*.cpp")
    add_includedirs("include")
    set_rundir("$(projectdir)")
    add_packages("openmp")
    add_cxxflags("-std=c++20 -fopenmp -O3")

