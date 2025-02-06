# VMEC++ large tests

Suite of VMEC++ C++ tests that require large test files which we prefer not
to include in the [standalone VMEC++ repo](https://github.com/proximafusion/vmecpp).

In order to run these tests, clone them inside the VMEC++ repo:

```
# skip if you have vmecpp already cloned
git clone --recurse-submodules git@github.com:proximafusion/vmecpp
cd vmecpp/src/vmecpp/cpp
git clone git@github.com:proximafusion/vmecpp_large_cpp_tests

# build and run all tests
bazel test --config=opt //vmecpp/... //vmecpp_large_cpp_tests/...
```
