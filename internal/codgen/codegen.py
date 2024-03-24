import argparse
import os 

class MetaDatabase(object):
    def __init__(self):
        pass
    
    def generate_impl_begin(self):
        return "Hello"

    def generate_impl_end(self):
        return "World"

    def write(self, path, content):
        open(path, "wb").write(content.encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate code from meta files")
    # parser.add_argument("generators", help="generator file list", nargs="*")
    parser.add_argument("--outdir", help="output directory", required=True, type=str) 
    args = parser.parse_args()
    db = MetaDatabase()

    impl_content: str = db.generate_impl_begin()
    impl_content += " "
    impl_content += db.generate_impl_end()

    db.write(os.path.join(args.outdir, "generated.cpp"), impl_content)
