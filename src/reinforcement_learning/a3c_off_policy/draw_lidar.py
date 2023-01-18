import numpy as np
import matplotlib.pyplot as plt
import torch
from tf.transformations import euler_from_quaternion, quaternion_from_euler

lidar = [0.18417638540267944, 0.18469049036502838, 0.18422454595565796, 0.194715678691864, 0.18473656475543976, 0.1785355508327484, 0.1837720423936844, 0.1760275661945343, 0.18163242936134338, 0.17476145923137665, 0.1662742644548416, 0.1745551973581314, 0.1782335788011551, 0.18549133837223053, 0.1818346232175827, 0.18462441861629486, 0.18437032401561737, 0.1869988888502121, 0.16071371734142303, 0.1849018782377243, 0.16876305639743805, 0.1824675351381302, 0.1677730828523636, 0.16780243813991547, 0.18324856460094452, 0.185384601354599, 0.16056974232196808, 0.17115670442581177, 0.18222689628601074, 0.17385941743850708, 0.16998204588890076, 0.17576342821121216, 0.1695195734500885, 0.17590616643428802, 0.19872000813484192, 0.18425749242305756, 0.17348536849021912, 0.1574161797761917, 0.17044654488563538, 0.1776525378227234, 0.17905117571353912, 0.18612152338027954, 0.19144079089164734, 0.1997046023607254, 0.18288534879684448, 0.184138223528862, 0.1871553361415863, 0.20230841636657715, 0.19101493060588837, 0.17866985499858856, 0.18640443682670593, 0.19961866736412048, 0.19530394673347473, 0.18688470125198364, 0.1932268738746643, 0.1950688660144806, 0.19080594182014465, 0.1965779960155487, 0.21071022748947144, 0.20954293012619019, 0.19913704693317413, 0.23531591892242432, 0.20988501608371735, 0.21151074767112732, 0.22796355187892914, 0.2252596914768219, 0.22642254829406738, 0.22378863394260406, 0.23747655749320984, 0.24412082135677338, 0.25167232751846313, 0.24205368757247925, 0.25806814432144165, 0.2696208357810974, 0.261290967464447, 0.27682948112487793, 0.27592670917510986, 0.29827263951301575, 0.2873302400112152, 0.30502867698669434, 0.32783183455467224, 0.3029592037200928, 0.32971271872520447, 0.3331696093082428, 0.34463268518447876, 0.3464677333831787, 0.34611162543296814, 0.3764779269695282, 0.3688378930091858, 0.3979530334472656, 0.4154074490070343, 0.43398138880729675, 0.4539061486721039, 0.490421861410141, 0.5016574263572693, 0.5094099640846252, 0.5717655420303345, 0.5935308933258057, 0.6143375635147095, 0.6601408123970032, 0.699320375919342, 0.7656545042991638, 0.8021823167800903, 0.9018878936767578, 0.9708184003829956, 1.062153935432434, 1.189609408378601, 1.3489280939102173, 1.595538854598999, 1.8727725744247437, 2.2983036041259766, 2.683988332748413, 2.6990396976470947, 2.6933584213256836, 2.686028242111206, 2.704087972640991, 2.6839468479156494, 2.680135726928711, 2.698638439178467, 2.6940834522247314, 2.7033419609069824, 2.7160089015960693, 2.7119462490081787, 2.746457576751709, 2.732638120651245, 2.7578959465026855, 2.7482213973999023, 2.755126714706421, 2.780613422393799, 2.805696487426758, 2.7792723178863525, 2.79656982421875, 2.8344688415527344, 2.857720136642456, 2.8526289463043213, 2.882171392440796, 2.9117038249969482, 2.918438196182251, 2.9388933181762695, 2.983325958251953, 2.988939046859741, 3.0240046977996826, 3.044262170791626, 3.074204444885254, 3.096994161605835, 3.1383116245269775, 3.1852784156799316, 3.1939077377319336, 3.2302470207214355, 3.2608261108398438, 3.3046584129333496, 3.350304126739502, 3.4228508472442627, 3.464920997619629, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.262216091156006, 3.2153093814849854, 3.184537649154663, 3.176555871963501, 3.139073610305786, 3.1054019927978516, 3.0923962593078613, 3.067652940750122, 3.0421488285064697, 3.020369529724121, 2.9883792400360107, 2.9906702041625977, 2.959731340408325, 2.962545394897461, 2.9411368370056152, 2.9325528144836426, 2.8913156986236572, 3.0575108528137207, 3.310103416442871, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 0.8716452717781067, 0.8648485541343689, 0.8918011784553528, 0.9055083394050598, 0.8905605673789978, 0.8994172811508179, 0.9035224914550781, 0.9127182364463806, 0.9232925772666931, 0.9334378838539124, 0.9529651999473572, 0.9472481608390808, 0.9557681679725647, 0.9732589721679688, 0.9978548884391785, 0.9951074123382568, 1.0065596103668213, 1.0259901285171509, 1.0228078365325928, 1.034668207168579, 1.0673460960388184, 1.0757172107696533, 1.1009924411773682, 1.1395180225372314, 1.1310285329818726, 1.1614058017730713, 1.1657302379608154, 1.1958088874816895, 1.2149142026901245, 1.2541592121124268, 1.2717602252960205, 1.2867158651351929, 1.3278504610061646, 1.3735365867614746, 1.387695550918579, 1.3933805227279663, 1.4503631591796875, 1.4984489679336548, 2.706380605697632, 2.6846048831939697, 2.649925947189331, 2.620192050933838, 2.60079288482666, 2.5879757404327393, 2.5591859817504883, 2.5433599948883057, 2.501347303390503, 2.4887237548828125, 2.484727144241333, 2.4731807708740234, 2.437514305114746, 2.441843032836914, 2.419865369796753, 2.3866395950317383, 2.3716437816619873, 2.3681483268737793, 2.372128963470459, 2.38250994682312, 2.3492283821105957, 2.3606061935424805, 2.306004285812378, 2.331185817718506, 2.3102059364318848, 2.312760829925537, 2.30889630317688, 2.3196396827697754, 2.315770149230957, 2.2871079444885254, 2.289121150970459, 2.294412612915039, 2.297729730606079, 2.2949624061584473, 2.2810192108154297, 2.29611873626709, 2.2939906120300293, 1.911997675895691, 1.6046717166900635, 1.3941696882247925, 1.2227108478546143, 1.0967167615890503, 0.9964655041694641, 0.8990607857704163, 0.8018220067024231, 0.7675256133079529, 0.7108446359634399, 0.67188560962677, 0.6047703623771667, 0.57912677526474, 0.544424831867218, 0.5473695397377014, 0.49206188321113586, 0.4792468845844269, 0.47218334674835205, 0.42766284942626953, 0.4074573218822479, 0.3930519223213196, 0.37784916162490845, 0.3702252507209778, 0.36225008964538574, 0.3426670432090759, 0.3422304391860962, 0.3284560441970825, 0.31453827023506165, 0.31095385551452637, 0.31349679827690125, 0.29449090361595154, 0.30502745509147644, 0.30326083302497864, 0.2733558714389801, 0.26602232456207275, 0.27697980403900146, 0.2696390450000763, 0.24637246131896973, 0.24888509511947632, 0.2471826672554016, 0.24245016276836395, 0.24064970016479492, 0.23039352893829346, 0.21411889791488647, 0.23288194835186005, 0.232640340924263, 0.2176051288843155, 0.23128291964530945, 0.20879124104976654, 0.2027849704027176, 0.20036503672599792, 0.21364450454711914, 0.19983552396297455, 0.2106504738330841, 0.20019149780273438, 0.19455917179584503, 0.20637330412864685, 0.20203544199466705, 0.20999519526958466, 0.20004388689994812, 0.18746423721313477]

data = torch.zeros(360,2)
quaternions = [0,0,-0.84,0.5399]

_,_,yaw = euler_from_quaternion(quaternions)
print(yaw * 180 / torch.pi)

for i in range(360):
    if lidar[i] > 3.5:
        lidar[i] = 3.5
    data[i,0] = lidar[i] * np.cos((i) * torch.pi / 180)
    data[i,1] = lidar[i] * np.sin((i) * torch.pi / 180)

plt.scatter(data[:,0].cpu().numpy(), data[:,1].cpu().numpy())
plt.show()