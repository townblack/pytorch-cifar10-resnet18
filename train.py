import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import time
from tensorboardX import SummaryWriter

'''
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''
# 定义是否使用GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 240   #遍历数据集次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.001        #学习率
Milestones=[135,185]
Debug=False

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=True, download=False, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=Milestones,gamma = 0.1)

writer=SummaryWriter("./logs")

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(EPOCH):
                train_loss=0.0
                train_accu=0.0
                val_loss=0.0
                val_accu=0.0

                scheduler.step()
                #print(type(optimizer.param_groups[0]))
                #print(optimizer.param_groups[0]["lr"])
                #print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0
                begin=time.time()
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    if Debug:
                        print("[Epoch:{}/{}, Batch:{}/{}] Loss: {:.3f} | Acc: {:.3f}%".format(epoch+1,EPOCH,i+1,int(trainset.__len__()/BATCH_SIZE),sum_loss/(i+1),100.*correct/total))
                    
                    f2.write("[Epoch:{}/{}, Batch:{}/{}] Loss: {:.3f} | Acc: {:.3f}%".format(epoch+1,EPOCH,i+1,int(trainset.__len__()/BATCH_SIZE),sum_loss/(i+1),100.*correct/total))
                    f2.write('\n')
                    f2.flush()

                train_loss=sum_loss/int(trainset.__len__()/BATCH_SIZE)
                train_accu=100.*correct/total

                # 每训练完一个epoch测试一下准确率
                with torch.no_grad():
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                        sum_loss += loss.item()
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    
                val_loss=sum_loss/int(testset.__len__()/BATCH_SIZE)
                val_accu = 100.*correct/total
                end=time.time()
                print("[Epoch:{}/{}] Train Loss: {:.3f} | Train Acc: {:.3f}% Test Loss: {:.3f} | Test Acc: {:.3f}% Cost time:{:.2f}min".format(epoch+1,EPOCH,train_loss,train_accu,val_loss,val_accu,(end-begin)/60.0))
                
                writer.add_scalar("Loss/train",train_loss,epoch)
                writer.add_scalar("Loss/val",val_loss,epoch)
                writer.add_scalar("Accu/train",train_accu,epoch)
                writer.add_scalar("Accu/val",val_accu,epoch)
                writer.add_scalar("Learning rate",optimizer.param_groups[0]["lr"],epoch)

                # 将每次测试结果实时写入acc.txt文件中
                #print('Saving model......')
                torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf,epoch + 1))
                f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch+1,val_accu))
                f.write('\n')
                f.flush()
                # 记录最佳测试分类准确率并写入best_acc.txt文件中
                if val_accu > best_acc:
                    f3 = open("best_acc.txt", "w")
                    f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch+1,val_accu))
                    f3.close()
                    best_acc = val_accu

            print("Training Finished, TotalEPOCH=%d" % EPOCH)
