from ptflops import get_model_complexity_info
from models import create_mobilevit_model, mobilevit_mini_cfg


def main():
    net = create_mobilevit_model(mobilevit_mini_cfg).cuda()
    # net = create_efficientvit_model(efficientvit_b3_cfg).cuda()
    print('Model Loaded! Calculating Flops...')
    macs, params = get_model_complexity_info(net, 
                                                input_res=(3, 224, 224), 
                                                as_strings=True,
                                                print_per_layer_stat=True, 
                                                verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == "__main__":
    main()
