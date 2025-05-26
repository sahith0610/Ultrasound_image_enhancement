class Ultrasounddataset(Dataset):
    def __init__(self, rootA, rootB, size):
        exts = ('*.jpg','*.jpeg','*.png')
        self.pathsA = sum([glob.glob(os.path.join(rootA, '**', e), recursive=True)
                            for e in exts], [])
        self.pathsB = sum([glob.glob(os.path.join(rootB, '**', e), recursive=True)
                            for e in exts], [])
        self.pathsA.sort(); self.pathsB.sort()
        self.size = size

    def __len__(self):
        return max(len(self.pathsA), len(self.pathsB))

    def __getitem__(self, idx):
        pathA = self.pathsA[idx % len(self.pathsA)]
        pathB = self.pathsB[idx % len(self.pathsB)]
        imgA = Image.open(pathA).convert('L').resize((self.size, self.size))
        imgB = Image.open(pathB).convert('L').resize((self.size, self.size))
        tf = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        return tf(imgA), tf(imgB)