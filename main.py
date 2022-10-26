from model import PopMusicTransformer
import os
# default to use CPU (-1)
# To use GPU, set the value below to 0, 1, 2, ...
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-tempo-checkpoint',
        is_training=False)

    # generate from scratch
    # model.generate(
    #    n_target_bar=16,
    #    temperature=1.2,
    #    topk=5,
    #    output_path='./result/from_scratch.midi',
    #    prompt=None)

    # generate continuation
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/continuation.midi',
        prompt='./data/evaluation/000.midi')

    # close model
    model.close()


if __name__ == '__main__':
    main()
