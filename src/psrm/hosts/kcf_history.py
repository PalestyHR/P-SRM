import numpy as np

def causal(frame,candidate,margin,score,valid,query_frame=0):
    previous=[];prediction_error_sq=[];consecutive=0;features=[];ids=[]
    for position,current_frame in enumerate(frame):
        current_xy=candidate[position]
        if valid[position]:
            if len(previous)>=2:
                p1,p2=previous[-2:];dt=max(int(frame[p2]-frame[p1]),1)
                velocity=(candidate[p2]-candidate[p1])/float(dt)
                predicted=candidate[p2]+velocity*float(current_frame-frame[p2])
                prediction_error_sq.append(float(np.sum((current_xy-predicted)**2)))
            previous.append(position);consecutive=0;continue
        row=np.zeros(21,np.float32);age=max(int(current_frame)-query_frame,1)
        row[2]=age;row[3]=len(previous);row[4]=len(previous)/float(age);row[6]=consecutive
        if previous:
            last=previous[-1];gap=int(current_frame-frame[last]);delta=current_xy-candidate[last]
            row[0]=1;row[5]=gap;row[7]=margin[last];row[8]=score[last]
            row[9:11]=delta;row[11]=float(np.linalg.norm(delta))
            if len(previous)>=2:
                before=previous[-2];dt=max(int(frame[last]-frame[before]),1)
                velocity=(candidate[last]-candidate[before])/float(dt)
                predicted=candidate[last]+velocity*float(gap);innovation=current_xy-predicted
                row[1]=1;row[12:14]=velocity;row[14]=float(np.linalg.norm(velocity))
                row[15:17]=innovation;row[17]=float(np.linalg.norm(innovation))
                row[20]=float(margin[last]-margin[before])/float(dt)
                if prediction_error_sq:
                    rms=float(np.sqrt(np.mean(prediction_error_sq)));row[18]=rms
                    row[19]=row[17]/max(rms,1e-3)
        features.append(row);ids.append(int(current_frame));consecutive+=1
    return np.asarray(ids,np.int32),np.asarray(features,np.float32).reshape(-1,21)
