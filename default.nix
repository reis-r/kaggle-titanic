# This imports the nix package collection,
# so we can access the `pkgs` and `stdenv` variables
with import <nixpkgs> {};

# Make a new "derivation" that represents our shell
stdenv.mkDerivation {
  name = "hy";

  # The packages in the `buildInputs` list will be added to the PATH in our shell
  buildInputs = let
    hy = python3.pkgs.buildPythonPackage rec {
      pname = "hy";
      version = "0.18.0";
      src = python3.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "42f24caaa7f5b4427929859395c215c8cc6e19807b46feaa0b863f3346e5ae11";
      };
      doCheck = false;
      propagatedBuildInputs = with python3Packages; [
        appdirs
        astor
        clint
        fastentrypoints
        funcparserlib
        rply
        colorama
      ];
      meta = {
        description = "A LISP dialect embedded in Python";
        homepage = "http://hylang.org/";
      };
    };
  in [
    # see https://nixos.org/nixos/packages.html to search for more
    hy
    python37Packages.Keras
    python37Packages.pandas
    python37Packages.matplotlib
    python37Packages.tensorflow
    python37Packages.scikitlearn
  ];

  shellHook = ''
            hy titanic.hy && exit
  '';
}
