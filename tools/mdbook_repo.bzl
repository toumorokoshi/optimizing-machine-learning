def _mdbook_repo_impl(repository_ctx):
    arch = repository_ctx.os.arch
    os_name = repository_ctx.os.name.lower()
    
    # Default to Linux x86_64
    url = "https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-x86_64-unknown-linux-gnu.tar.gz"
    sha = "9ef07fd288ba58ff3b99d1c94e6d414d431c9a61fdb20348e5beb74b823d546b"
    
    if "mac os" in os_name or "os x" in os_name:
        url = "https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-x86_64-apple-darwin.tar.gz"
        sha = "9eb3e82d029d5b069d300bbf38cb441865c192c77d54ea3d51eb923187c53d08"
    elif "linux" in os_name:
        if arch == "aarch64" or arch == "arm64":
            url = "https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-aarch64-unknown-linux-musl.tar.gz"
            sha = "d5ebfd2e31755726f8c0988a423b1ab5d950bb894fbba170355bb7d1cb852586"
        else:
            url = "https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-x86_64-unknown-linux-gnu.tar.gz"
            sha = "9ef07fd288ba58ff3b99d1c94e6d414d431c9a61fdb20348e5beb74b823d546b"
            
    repository_ctx.download_and_extract(
        url = url,
        sha256 = sha,
    )
    
    # Create the BUILD file inside the repository to expose the binary
    repository_ctx.file(
        "BUILD",
        """
filegroup(
    name = "bin",
    srcs = ["mdbook"],
    visibility = ["//visibility:public"],
)
"""
    )

mdbook_repo = repository_rule(
    implementation = _mdbook_repo_impl,
)
