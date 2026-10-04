import mvt_engine as rt


@rt.method()
def normalize(language, texts):
    return [list(text.replace(" ", "")) for text in texts]


if __name__ == "__main__":
    rt.run()
