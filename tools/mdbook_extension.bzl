load("//tools:mdbook_repo.bzl", "mdbook_repo")

def _mdbook_ext_impl(module_ctx):
    mdbook_repo(name = "mdbook_bin")

mdbook_ext = module_extension(
    implementation = _mdbook_ext_impl,
)
