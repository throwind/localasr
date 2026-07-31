from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


module_collection_mode = "pyz+py"
datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
        "**/include/**",
        "**/testing/**",
        "**/test/**",
        "**/tests/**",
    ],
)
binaries = collect_dynamic_libs("torch")
hiddenimports = []
