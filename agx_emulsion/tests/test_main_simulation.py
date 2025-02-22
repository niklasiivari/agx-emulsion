import matplotlib.pyplot as plt
from agx_emulsion.model.process import photo_params, photo_process
from agx_emulsion.utils.io import load_image_16bit_32bit

def test_main_simulation():
    image = load_image_16bit_32bit('img/targets/it87_test_chart_2.jpg')
    params = photo_params()
    params.negative.grain.sublayers_active = True
    params.io.preview_resize_factor = 1.0
    params.camera.exposure_compensation_ev = 2
    params.enlarger.print_exposure = 0.5
    params.camera.film_format_mm = 35
    params.print_paper.glare.active = True
    print_scan = photo_process(image, params)
    params.io.compute_negative = True
    negative_scan = photo_process(image, params)

    _, axs = plt.subplots(1,2)
    axs[0].imshow(negative_scan)
    axs[0].axis('off')
    axs[0].set_title('negative')
    axs[1].imshow(print_scan)
    axs[1].axis('off')
    axs[1].set_title('print')
    plt.show()
    return

if __name__ == '__main__':
    test_main_simulation()