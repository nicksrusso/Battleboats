# One test executable per component, mirroring the build stages. For now these
# are STUBS: each test_*.cpp holds only `int main() { return 0; }`, so they
# link the engine and exit clean. That's deliberate — stage 0 proves the whole
# configure -> build -> ctest loop works before there's any logic to test.
#
# When we reach real assertions we'll pull in a test framework (Catch2 or
# doctest) via FetchContent and replace these bodies. The structure here won't
# change.

# A small helper so we don't repeat the three-line pattern for every test.
# Given a name, it builds test_<name> from test_<name>.cpp, links the engine,
# and registers it with CTest under <name>.
function(add_engine_test name)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE engine)
    add_test(NAME ${name} COMMAND test_${name})
endfunction()

add_engine_test(ship_data)
add_engine_test(map)
add_engine_test(game_engine)
