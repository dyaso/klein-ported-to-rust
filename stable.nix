# Use latest nightly Rust
#
# To use, clone the Mozilla overlay, and provide the path at nix-shell
# invocation, e.g.:
#
#     git clone https://github.com/mozilla/nixpkgs-mozilla.git
#     nix-shell nightly.nix -I rustoverlay=/path/to/overlay
#
# If you want to update Rust versions to the nightly builds, just pull the
# overlay repository.

with import <nixpkgs> {};
with import <rustoverlay/rust-overlay.nix> pkgs pkgs;
# nix-shell stable.nix -I rustoverlay=~/languages/rust/nixpkgs-mozilla

#https://eipi.xyz/blog/rust-overlay-nix-shell/
#i put the following at the end of `/etc/nixos/configuration.nix` :
  # programs.bash.shellAliases = {
  #   ns = "nix-shell stable.nix -I rustoverlay=~/languages/rust/nixpkgs-mozilla";
  # };

stdenv.mkDerivation {
  name = "rust-env";
  nativeBuildInputs = [
  	    libxkbcommon

        # for druid
        cairo
        pango
        atk
        gdk-pixbuf
        gtk3-x11

        #needed for `shello` example
        #glib

  	    pkgconfig
  	    x11
];
  buildInputs = [
    latest.rustChannels.stable.rust
    #latest.rustChannels.nightly.rust
  	    xorg.libXi
  	    xorg.libXrandr
  	    xorg.libXcursor
];

  RUST_BACKTRACE = 1;


  }

# stdenv.mkDerivation {
#   name = "url-bot-rs";

#   buildInputs = [
#     latest.rustChannels.nightly.rust
#     pkgconfig
#     openssl
#     sqlite
#   ];
# }
