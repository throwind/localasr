from PyInstaller.utils.hooks import collect_dynamic_libs


module_collection_mode = "pyz+py"
binaries = collect_dynamic_libs("torchaudio")
hiddenimports = ["torchaudio.lib", "torchaudio.lib._torchaudio", "torchaudio.lib.libtorchaudio"]
