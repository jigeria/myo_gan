hand_image_names = []

data = pd.read_csv('./Sample_data/emg_data.csv', sep=',')
data = pd.read_csv('./Sample_data/hand_images/time_match.csv')
data = data.values.tolist()
data = np.asarray(data)

image_number = 0

for dirname, dirnames, filenames in os.walk('./Sample_data/hand_images'):
    image_number = len(filenames)

for i in range(len(data)):
    data[i][0] = int(i * (image_number / len(data)))

np.savetxt('./time_match.csv', data.astype(float), delimiter=',')