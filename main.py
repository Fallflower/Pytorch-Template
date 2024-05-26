from train import train
from models.model_info import export_model
from assistant import Options, HelpSave


def main(model):
    options = Options()
    opt = options.load_options(f'./models/{model}/options.json')
    saver = HelpSave()
    saver.init_save_dir(opt)
    train(opt, saver)


if __name__ == '__main__':
    # main("VGG19")
    # main("DenseNet121")
    # main("VisionTransformer")
    export_model("DenseNet121")
