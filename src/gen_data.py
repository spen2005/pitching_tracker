import os
import numpy as np
import pandas as pd

def rotation_matrix(theta,phi):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    matrix1 = np.array([[cos_theta, -sin_theta, 0],
                        [sin_theta, cos_theta, 0],
                        [0, 0, 1]])
    
    matrix2 = np.array([[cos_phi, 0, -sin_phi],
                        [0, 1, 0],
                        [sin_phi, 0, cos_phi]])

    return matrix2.dot(matrix1)

def camera(vector):
    x = vector[0]
    y = vector[1]
    z = vector[2]
    theta = np.arctan2(y,x)
    phi = np.pi-np.arctan2(np.sqrt(x**2 + y**2),z)
    return rotation_matrix(-theta,-phi)

def position_after_transformation(cam_pos,cam_view,point):
    point = point - cam_pos
    matrix = camera(cam_view)
    return matrix.dot(point)

def generate_data(n):
    # input data (vec1,vec2,...,vec5,timestamp,x',y',x,y,z)
    # 5 fixed points as information source
    dataset = []
    for i in range(n):
        # camera position (x,y,z)=(100~130,20~35,5~20) randomly sample
        # camera view (x-10~x+10,y-10~y+10,z-10~z+10) randomly sample
        x = np.random.uniform(135,170)
        y = np.random.uniform(0,20)
        z = np.random.uniform(2,20)
        cam_pos = np.array([x,y,z])
        x = np.random.uniform(x-10,x+10)
        y = np.random.uniform(y-10,y+10)
        z = np.random.uniform(z-10,z+10)
        cam_view = np.array([x,y,z])
        #normalize
        cam_view = cam_view/np.linalg.norm(cam_view)
        # fixed points
        vec1 = np.array([-0.2164,0.0,0.0])
        vec2 = np.array([0.91,0.365,0.0])
        vec3 = np.array([-0.91,0.365,0.0])
        vec4 = np.array([0.91,-0.365,0.0])
        vec5 = np.array([-0.91,-0.365,0.0])
        # transforamtion
        vec = position_after_transformation(cam_pos,cam_view,vec1)
        vec1_prime = np.array([vec[0]/vec[2],vec[1]/vec[2]])
        

        vec = position_after_transformation(cam_pos,cam_view,vec2)
        vec2_prime = np.array([vec[0]/vec[2],vec[1]/vec[2]])
        vec2_prime = vec2_prime-vec1_prime
        regularization_value = np.linalg.norm(vec2_prime)
        vec2_prime = vec2_prime/regularization_value

        vec = position_after_transformation(cam_pos,cam_view,vec3)
        vec3_prime = np.array([vec[0]/vec[2],vec[1]/vec[2]])
        vec3_prime = vec3_prime-vec1_prime
        vec3_prime = vec3_prime/regularization_value

        vec = position_after_transformation(cam_pos,cam_view,vec4)
        vec4_prime = np.array([vec[0]/vec[2],vec[1]/vec[2]])
        vec4_prime = vec4_prime-vec1_prime
        vec4_prime = vec4_prime/regularization_value

        vec = position_after_transformation(cam_pos,cam_view,vec5)
        vec5_prime = np.array([vec[0]/vec[2],vec[1]/vec[2]])
        vec5_prime = vec5_prime-vec1_prime
        vec5_prime = vec5_prime/regularization_value

        x = np.random.uniform(17.44,18.44)
        y = np.random.uniform(-1.5,1.5)
        z = np.random.uniform(1.6,2.4)
        start_point = np.array([x,y,z])
        x = 0
        y = np.random.uniform(-0.6,0.6)
        z = np.random.uniform(0,2.1)
        end_point = np.array([x,y,z])
        # segment (start,end) cut into 10 pieces 
        for t in range (0,10):
            timestamp = t
            tt = np.random.uniform(0,10)
            timestamp = tt
            x = start_point[0]+tt*(end_point[0]-start_point[0])/9
            y = start_point[1]+tt*(end_point[1]-start_point[1])/9
            z = start_point[2]+tt*(end_point[2]-start_point[2])/9
            point = np.array([x,y,z])
            point_prime = position_after_transformation(cam_pos,cam_view,point)
            point_prime = np.array([point_prime[0]/point_prime[2],point_prime[1]/point_prime[2]])
            point_prime = point_prime-vec1_prime
            #add to the dataset
            data = []
            # 取小數點後3位
            vec2_prime = np.round(vec2_prime,2)
            vec3_prime = np.round(vec3_prime,2)
            vec4_prime = np.round(vec4_prime,2)
            vec5_prime = np.round(vec5_prime,2)
            data.append(vec2_prime[0])
            data.append(vec2_prime[1])
            data.append(vec3_prime[0])
            data.append(vec3_prime[1])
            data.append(vec4_prime[0])
            data.append(vec4_prime[1])
            data.append(vec5_prime[0])
            data.append(vec5_prime[1])
            #data.append(vec6_prime[0])
            #data.append(vec6_prime[1])
            #data.append(vec7_prime[0])
            #data.append(vec7_prime[1])
            #data.append(vec8_prime[0])
            #data.append(vec8_prime[1])
            data.append(timestamp)
            # normalize
            point_prime = point_prime/regularization_value
            # 取小數點後兩位
            point = 100*point

            point_prime = np.round(point_prime,2)
            point = np.round(point,2)
            cam_pos = np.round(cam_pos,2)

            data.append(point_prime[0])
            data.append(point_prime[1])
            data.append(point[0])
            data.append(point[1])
            data.append(point[2])
            #data.append(cam_pos[0])
            #data.append(cam_pos[1])
            #data.append(cam_pos[2])
            dataset.append(data)
        if i % 100000 == 0:
            print(i/100000)
            print(n/100000)
    #randomly sort the dataset then return
    np.random.shuffle(dataset)
    return dataset

# 获取当前脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 确保目录存在，否则创建目录
data_dir = os.path.join(script_dir, "../data")
os.makedirs(data_dir, exist_ok=True)

# 生成训练数据
train_data = generate_data(100)
# 将数据保存到指定路径下
train_file = os.path.join(data_dir, "train_data.csv")
df = pd.DataFrame(train_data)
df.to_csv(train_file, index=False)
print('train data generated')

# 生成测试数据
test_data = generate_data(50000)
# 将数据保存到指定路径下
test_file = os.path.join(data_dir, "test_data.csv")
df = pd.DataFrame(test_data)
df.to_csv(test_file, index=False)
print('test data generated')

    


        


        
