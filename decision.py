import numpy as np
def decision(result,thresh=2):
    pos=np.argwhere(result>0.5).flatten()
    rpos = []
    pre = 0
    last = len(pos)
    for j in np.where(np.diff(pos)>2)[0]:
        if j-pre>thresh:
            rpos.append((pos[pre]+pos[j])*4)
        pre = j+1
    rpos.append((pos[pre]+pos[last-1])*4)
    qrs = np.array(rpos)
    qrs_diff = np.diff(qrs)
    check = True
    while check:
        qrs_diff = np.diff(qrs)
        if len(qrs_diff)>1:
            for r in range(len(qrs_diff)):
                if qrs_diff[r]<100:
                    if result[int(qrs[r]/8)]>result[int(qrs[r+1]/8)]:
                        qrs = np.delete(qrs,r+1)
                        check = True
                        break
                    else:
                        qrs = np.delete(qrs,r)
                        check = True
                        break
                check = False
        else:
            check = False
    return qrs

def recheck(result,qrs,thresh=1):
    qrs_diff = np.diff(qrs)
    miss = np.where(qrs_diff>600)[0]
    for i in miss:
        add_qrs = decision(result[qrs[i]//8-1:qrs[i+1]//8+2],thresh=thresh)
        if len(add_qrs)>2:
            maxposb = add_qrs[1]
            for add in add_qrs[1:-1]:
                if result[qrs[i]//8-1+add//8]>result[qrs[i]//8-1+maxposb//8]:
                    maxposb = add
            qrs = list(qrs)
            qrs.append(maxposb+qrs[i]-8)
    qrs = np.sort(np.array(qrs))
    return qrs

def QRS_decision(result):
    qrs = decision(result,thresh=2)
    qrs_diff = np.diff(qrs)
    if np.max(qrs_diff)>600:
        qrs = recheck(result,qrs,thresh=1)
        qrs_diff = np.diff(qrs)
        if np.max(qrs_diff)>600:
            qrs = recheck(result,qrs,thresh=0)
            qrs_diff = np.diff(qrs)
            if np.max(qrs_diff)>600:
                qrs = recheck(result,qrs,thresh=-1)
    return qrs
    
def performance(data,refs,model,write_to_file=False,name=None,fs=500):
    R_ans = []
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for i in range(len(data)):
        pred = model.predict(data[i].reshape(1,-1,1))[0][:,0]
        r_ans = QRS_decision(pred)
        if name and name[i] == '207': #remove predictions from flutter segments of record 207 in MITDB 
            r_ans = r_ans[~((r_ans>14894//360*500-250)*(r_ans<21608//360*500+250))]
            r_ans = r_ans[~((r_ans>87273//360*500-250)*(r_ans<100956//360*500+250))]
            r_ans = r_ans[~((r_ans>554826//360*500-250)*(r_ans<589660//360*500+250))]
        
        R_ans.append(r_ans)
        if write_to_file:
            if name:
                filename = name[i]
            else:
                filename = str(i)
            np.savetxt('./output/'+filename+'_QRS.txt',r_ans*fs//500,fmt='%d',delimiter=' ')

        if refs:
            tp = 0
            fp = 0
            fn = 0
            for j in refs[i]:
                loc = np.where(np.abs(R_ans[i]-j)<500*0.15)[0]
                if len(loc)>0:
                    tp += 1
                    fp += len(loc)-1
                else:
                    fn += 1
            for k in R_ans[i]:
                loc = np.where(np.abs(refs[i]-k)<500*0.15)[0]
                if len(loc) == 0:
                    fp += 1
            tp_all += tp
            fp_all += fp
            fn_all += fn
            if name is None:
                record = str(i)
            else:
                record = name[i]
            with open('./output/results.txt','a') as f:
                f.write('%10s %10d %10d %10d\n'%(record,tp,fp,fn))
            print('%10s %10d %10d %10d'%(record,tp,fp,fn))
    return tp_all,fp_all,fn_all