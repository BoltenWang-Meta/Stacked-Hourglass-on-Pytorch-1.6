from    Hyper import Hyper
from    Evaluate.Inference import ShowSamples

def main():
    net, hyper, LossFn, optimizer, train_loader, test_loader = Hyper(path)

    for epoch in range(hyper['ep']):
        print(epoch)
        for img, GT in train_loader:
            net.train()
            img, GT = img.to(hyper['dv']), GT.to(hyper['dv'])

            map_list = net(img)

            temp_loss = []
            for map in map_list:
                temp_loss.append(LossFn(map, GT))
            loss = sum(temp_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ShowSamples(net)

if __name__ == '__main__':
    path = r"E:\9_A_PhD\DataSet\Leeds_Sport_Pose\DataSet"
    main()

