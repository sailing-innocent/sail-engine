/**
 * @version
 * @brief   The subsystem module entry for sail engine runtime
 * @date    2023/12/09
 * @author sailing-innocent
 */

namespace sail {
class ModuleManager;
}  // namespace sail

namespace sail {

// a dependency specified in plugin.json
struct ModuleDependency {
  std::string name;     // the name of the dependency module
  std::string version;  // the version of the dependency module
};

// struct containing information about a module
struct ModuleInfo {};

}  // namespace sail