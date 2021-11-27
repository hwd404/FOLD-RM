from foldrm import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta

from algo import justify_one


def main():
    # model, data = acute()
    # model, data = autism()
    # model, data = breastw()
    # model, data = cars()
    # model, data = credit()
    # model, data = heart()
    # model, data = kidney()
    # model, data = krkp()
    # model, data = mushroom()
    # model, data = sonar()
    # model, data = voting()
    # model, data = ecoli()
    # model, data = ionosphere()
    # model, data = wine()
    # model, data = adult()
    # model, data = credit_card()
    # model, data = rain()
    model, data = heloc()

    data_train, data_test = split_data(data, ratio=0.8)

    start = timer()
    model.fit(data_train, ratio=0.9)
    end = timer()

    model.print_asp()
    Y_test_hat = model.predict(data_test)
    acc = get_scores(Y_test_hat, data_test)
    print('% acc', round(acc, 4))
    print('% foldrm costs: ', timedelta(seconds=end - start), '\n')

    exit()
    k = 1
    for i in range(len(data_test)):
        print('Explanation for example number', k, ':')
        model.explain(data_test[i])
        k += 1


if __name__ == '__main__':
    main()
