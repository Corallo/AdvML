def testNews(inputs, targets, kernel):
    print("Testing..")
    runs=10
    original_score=np.zeros(runs)
    kernel_score=np.zeros(runs)
    myrange=[2**x for x in range(1,9)] # test with different number of test_points (2,4,8,16 ..)

    for n in myrange: 
        print("\n\nUsing "+str(n)+" train points")
        for x in range(runs): #run 100 times
            zero = np.array(np.where(targets==0))[0]
            one =np.array( np.where(targets==1))[0]

            train_zero=np.array(np.random.choice(zero,int(n/2)))
            train_one =np.array(np.random.choice(one, int(n/2)))

            train_idx =np.concatenate((train_zero,train_one))


            np.random.shuffle(train_idx)
            #print(train_idx)
            train_targets = np.take(targets,train_idx)
            train_points = np.take(inputs,train_idx,axis=0)
            train_kernel = np.take(kernel, train_idx,axis=0)

            original = svm.SVC().fit(train_points, train_targets)
            original_score[x]=original.score(inputs, targets)
            
    
            new = svm.SVC(kernel='linear').fit(train_kernel, train_targets)
            kernel_score[x]=new.score(kernel, targets)

        
        print("Average score of normal SVM:")
        print(np.average(original_score))
        print("Average score of cluster kernel:")
        print(np.average(kernel_score))