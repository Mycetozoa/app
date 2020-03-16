import img_to_b64
# import neural

RESULT = 'test/result.txt'

with open(RESULT, 'w') as result:
    result.write(img_to_b64.encode('test/original.jpg'))

# neural.1_create_train_test.data_path = './test/'
