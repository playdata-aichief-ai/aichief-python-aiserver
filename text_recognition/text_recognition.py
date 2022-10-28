import string
import argparse

import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils.logging_time import logging_time

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate
from .model import Model
from ai.settings.settings import BASE_DIR

"""
CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
"""


class Text_Recognition():

    def __init__(self, Transformation='TPS', FeatureExtraction='ResNet', SequenceModeling='BiLSTM', Prediction='Attn', saved_model=os.path.join(BASE_DIR, 'text_recognition', 'saved_models', 'best_accuracy_lenplus_66.pth')):
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_folder',   # required=True,
                            help='path to image_folder which contains text images')
        parser.add_argument('--workers', type=int,
                            help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=500,  # training은 안되는데, predict는 가능. training때 32헀음
                            help='input batch size')  # 192
        parser.add_argument('--saved_model', default=saved_model,  # required=True,
                            help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int,
                            default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32,
                            help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100,
                            help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        # parser.add_argument('--character', type=str, default='!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~¨가각간갈감갑값갓갔강갖갗같개객갤갬갭갱걀걍거걱건걷걸검겁것겅겉게겍겐겔겟겠겨격겪견결겳겸겹겼경곁계고곡곤곧골곪곰곱곳공곷과곽관괄광괜괴교구국군굳굴굵굼굽굿궁궈권궤귀귄규귝균귤귬귭그극근귿글긁금급긋긍기긱긴길김깁깅깊까깍깎깐깔깝깥깨꺠꺼꺾껄껌껍껑께껴꼈꼉꼐꼬꼭꼼꼽꽂꽃꽈꽉꾸꾹꿀꿈꿍꿏꿔꿰뀌끄끈끊끌끓끔끗끝끼낀낄낌나낙난날남납낫났낭낮낯낱내낵낸낼냄냅냈냉냠냥너넉넌널넓넘넙넛넣네넥넨넬넴넵넷녀녁년녈념녕녛노녹논놀놈농높놓놔뇌뇨누눅눈눌눔눕눗눙뉘뉴뉸뉼늄늅느는늘능늦늬니닉닌닏닐님닙닛닝닡다닥닦단닫달닭닮담답닷당닻닽닿대댄댈댕댜더덕던덜덤덥덧덩덮데덱덴델뎀뎁뎌뎰도독돈돋돌돎돔돕돗동돞돠돼되된될됨됩됫두둑둔둘둠둡둣둥둪뒤뒷듀듈듐드득든듣들듬듭듯등듸디딕딘딜딤딥딩딪따딱딸땀땅때땐땡땨떄떠떡떤떨떻떼떽뗄뗴또똑똘똥뙤뚜뚝뚦뚫뚯뛰뜨뜩뜬뜯뜰뜸뜻띄띠띤라락란랄랆람랍랏랑랗래랙랜랠램랩랫랭략량러럭런럴럼럽럿렀렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례로록론롤롬롭롯롱롲롼뢰뢴료룔룡루룩룬룰룸룹룻룽뤀뤼류률륨륩르륵른를름릅릉릎릐릔리릭린릴림립릿링맄맇마막만많말맑맘맛망맞맟맡매맥맨맴맷맹맻먀먁먜머먹먼멀멁멈멋멍멎메멕멘멜며면멸명몇몔모목몬몰몸몹못몽뫼묘무묵문묺묻물묽묿뭂뭇뭉뮌뮤뮬뮴므믁믄믈믕미믹민믿밀밋밌밍밎및밑밒바박밖반받발밝밟밤밥방밭배백밴밸뱀뱅뱉뱍뱔버벅번벌범법벗벙벚베벡벤벨벳벼벽변별볍병볓볕보복본볼봄봅봇봉뵥부북분불붉붐붓붕붙붠뷰뷴뷸뷹븀브븐블븝비빅빈빌빕빗빙빛빠빨빼뺀뺏뺨뻐뻑뼈뽀뽑뽕뾰뿅뿌뿍뿐뿔뿜뿡쁘쁜쁠쁨삐삔삠사삭산살삶삼삽상샇새색샌샐샘생샤샨샬샴샵샷샹서석섞선설섬섭섯성세섹센셀셈셉셋셍셔션셜셨셩셰소속손솔솜솝송솦쇄쇠쇼쇽숄수숙순술숨숫숭숯숱숲쉐쉘쉬쉰쉴쉼쉽슈슌슐슘슙스슨슬슭슴습슷승슽시식신실심십싱싵싶싷싸싹싼쌀쌍쌓쌔쌕쌸써썩썬썸썹쎄쎼쏘쏙쏟쐐쑥쓰쓱쓴쓸씀씁씌씨씩씬씯씰씹씻씼씽아악안앉않알앏앓암압앗았앙앛앝앞애액앤앨앰앱앴앵야약앿얀얄얇얍양얖얗어억언얹얺얻얼얿엄업없엇었엉에엑엔엘엠엡엣엥여역연열엷엻염엽엿였영옅옆예옐옛오옥옧온올옮옴옵옷옹옻옿와왁완왔왕왜외왼요욕욘욜용욮우욱운울움웁웃웅워웍원월웠웨웬웰웹위윅윈윌윗유육윤율윰윳융으은을읋음읍응의읙읫이익인읻일읽잁잂잃임입잇있잉잊잋잌잍잎자작잔잘잝잠잡잣장잦재잭잴잼잿쟁쟈쟝저적전절젊점접젓정젖제젝젠젤젬젭젯젱젲젶져젹젼졌졔조족존졸좀좁종좋좌죠주죽준줄줅줌줍중줕줘쥐쥬쥰쥴즈즉즐즘즙증지직진짇질짐집짓징짙짚짛짜짝짠짧짱째쨰쩡쪼쪽쪾쫀쫓쬐쭈쭉쯔쯘찌찍찔찜찝찢차착찬찮찰참창찾채책챔챙챡처척천철첨첩첫청체첼쳐쳔초촉촌촐촘총촬최쵸추축춘출춞춤춥충춫춮춰췌취췸츄츌츠측츨층치칙친칠칡침칫칭칮카칸칼캀캄캅캐캔캘캠캡캴커컨컬컴컵컷케켄켈켐켑켓켚켜켠켰코콕콘콛콜콞콤콥콧콩콬콮콰콴쾌쿄쿠쿤쿨쿰쿳쿼퀄퀘퀴퀵퀸퀼큅큐큘크큭큰클큼킄키킥킨킬킴킵킷킹타탁탄탈탐탑탕태택탠탤탬탭탱탸터턱턴털텀텅테텍텐텔템텝텡토톡톤톨톰톱톳통톹퇴투툭툴툼퉁튀튈튜튤튬트특튼틀틈틍티틱틴틸팀팁팅팉파팍팎판팔팜팝팟팡팥패팩팬팸팹팻팽퍼펀펄펌펑페펙펜펠펨펩펴편펼폄평폐포폭폰폴폺폼퐁퐇표푸푹푼풀품풉풋풍퓌퓨퓰퓸프픈플픔픙피픽핀필핌핍핏핑하학한할핣핤함합핫항핮해핵핸핼햄햇했행햐햑햠햡향허헌헐험헙헛헝헤헥헨헬헴헵헷헹헿혀혁현혈혐협혓형혜호혹혼홀홈홉홍화확환활홤황회획횐횟횡효횬횽후훅훌훔훕훤훼휑휘휜휩휴흄흉흐흑흔흘흙흠흡흥흫희흰히힌힐힘힙＃％＆＇＊＋．：＜＝＞Ｘ［］＿｜～￣￦￨𝛃', help='character label')  # default='0123456789abcdefghijklmnopqrstuvwxyz'
        parser.add_argument('--character', type=str, default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~¨가각간갈감갑값갓갔강꼴갖갗같개객갤갬갭갱걀걍거걱건걷걸검겁것겅겉깜게겍겐겔겟겠겨격겪견결겳겸겹겼경곁계고곡곤곧골곪곰곱곳공곷과곽관괄광괜괴교구국군굳굴굵굼굽굿궁궈권궤귀귄규갚귝균귤귬귭그극근귿글긁금뺄급긋긍기긱긴길김깁깅깊까깍깎깐깔깝깥깨꺠꺼꺾껄껌껍껑께껴꼈꼉꼐꼬꼭꼼꼽꽂꽃꽈꽉꾸꾹꿀꿈꿍꿏꿔꿰뀌끄끈끊끌끓끔끗끝끼낀낄낌나낙난날남납낫났낭낮낯릇낱내낵낸낼냄냅냈냉냠냥너넉넌널넓넘넙넛넣네넥넨넬넴넵넷녀녁년녈념녕녛노녹논놀놈농높놓놔뇌뇨누눅눈눌눔눕눗눙뉘뉴뉸뉼늄늅느는늘능늦늬니닉닌닏닐님닙닛닝닡다닥닦단닫달닭닮담답닷당닻닽닿대댄댈댕댜더덕던덜덤덥덧덩덮데덱덴델뎀뎁뎌뎰도독돈돋돌돎돔돕돗동돞돠돼되된될됨됩됫두둑둔둘둠둡둣둥둪뒤뒷듀듈듐드득든듣들듬듭듯등듸디딕딘딜딤딥딩딪따딱딸땀땅때땐땡땨떄떠떡떤떨떻떼떽뗄뗴또똑똘똥뙤뚜뚝뚦뚫뚯뛰뜨뜩뜬뜯뜰뜸뜻띄띠띤라락란랄륙랆람랍랏랑랗래랙랜겆랠램랩랫랭략량러럭런럴럼럽럿렀렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례로록론롤롬롭롯롱롲롼뢰뢴료룔룡루룩룬룰룸룹룻룽뤀뤼류률륨륩르륵른를름릅릉릎릐릔리릭린릴림립릿링맄맇마막만많말맑맘맛망맞맟맡매맥맨맴맷맹맻먀먁먜머먹먼멀멁멈멋멍멎메멕멘멜며면멸명몇몔모목몬몰몸몹못몽뫼묘무묵문묺묻물묽묿뭂뭇뭉뮌뮤뮬뮴므믁믄믈믕미믹민믿밀밋밌밍밎및밑밒바박밖반받발밝밟밤밥방밭배백밴밸뱀뱅뱉뱍뱔버벅번벌범법벗벙벚베벡벤벨벳벼벽변별볍병볓볕보복본볼봄봅봇봉뵥부북분불붉붐붓붕붙붠뷰뷴뷸뷹븀브븐블븝비빅빈빌빕빗빙빛빠빨빼뺀뺏뺨뻐뻑뼈뽀뽑뽕뾰뿅뿌뿍뿐뿔뿜뿡쁘쁜쁠쁨삐삔삠사삭산살삶삼삽상샇새색샌샐샘생샤샨샬샴샵샷샹서석섞선설섬섭섯성세섹센셀셈셉셋셍셔션셜셨셩셰소속손솔솜솝송솦쇄쇠쇼쇽숄수숙순술숨숫숭숯숱숲쉐쉘쉬쉰쉴쉼쉽슈슌슐슘슙스슨슬슭슴습슷승슽시식신실심십싱싵싶싷싸싹싼쌀쌍쌓쌔쌕쌸써썩썬썸썹쎄쎼쏘쏙쏟쐐섰쑥쓰쓱쓴쓸씀씁씌씨씩씬씯씰씹씻씼씽아악안앉않알앏앓암압앗았앙앛앝깃앞애액앤앨앰앱앴앵야약앿얀얄얇얍양얖얗어억언얹얺얻얼얿엄업없엇었엉에엑엔엘엠엡엣엥여역연열엷엻염엽엿였영옅옆예옐옛냐뤄오옥옧온올옮옴옵옷옹옻옿와왁완왔왕왜외왼요욕욘욜용욮우욱운울움웁웃웅워웍원월웠웨웬웰웹위윅윈윌윗유육윤율윰윳융으은을읋음읍응의읙읫이익인읻일읽잁잂잃임입잇있잉짖잊잋잌잍잎자작잔잘잝잠잡잣장잦재잭잴잼잿맺쟁쟈쟝저적전절젊점접젓정젖제젝젠젤젬젭젯젱젲젶져젹젼졌졔조족존졸좀좁종좋좌죠주얘죽준줄줅줌줍중줕줘쥐쥬쥰쥴즈즉즐즘즙증지직진몫짇질짐집짓징짙짚썼짛짜짝짠쪘짧짱째쨰쩡쪼쪽쪾쫀쫓쬐쭈쭉쯔쯘찌찍찔찜찝찢차착찬찮찰참창찾채책싫챔챙묶챡처척봐천철첨첩첫청체첼쳐쳔초촉촌삿촐촘총촬최쵸추축춘출춞춤춥충춫춮춰췌취췸츄츌츠측츨층치칙친칠칡침칫칭칮카칸칼캀캄캅탓캐캔캘캠캡캴커컨컬컴컵컷케켄켈켐켑켓켚켜켠켰코콕콘콛콜콞콤콥콧콩콬콮콰콴쾌쿄쿠쿤쿨쿰쿳쿼퀄퀘퀴퀵퀸퀼큅큐큘크큭큰클큼킄키킥킨킬킴킵킷킹타탁탄탈탐탑탕태택탠탤탬탭탱탸터턱턴털텀텅테텍텐텔템텝텡토톡톤톨톰톱톳통톹퇴투툭툴툼퉁튀튈튜튤튬트특튼틀틈틍티틱틴틸팀팁팅팉파팍팎판팔팜팝팟팡팥패팩팬팸팹팻팽퍼펀펄펌펑페펙펜펠펨펩펴편펼폄평죄폐포폭폰폴폺폼퐁퐇표펭푸푹푼풀품풉풋풍퓌퓨퓰퓸프픈플픔픙피픽핀필핌핍핏핑하학한쳤할핣핤함합핫항핮해핵핸핼햄햇했행햐햑햠햡향허헌헐험헙헛헝헤헥헨헬헴헵헷헹헿혀혁현혈혐협혓형혜훈호혹혼홀홈홉홍화확환활홤황회획횐횟횡효횬횽후훅훌훔훕훤훼휑휘휜휩휴흄흉흐흑흔흘흙흠흡흥흫희흰히힌힐힘힙＃％＆＇＊＋．：＜＝＞Ｘ［］＿｜～￣￦￨𝛃궐륜띵눠랐옳빚괘낳둬봤빵눴듦찐꽤늑왈꼰곶뵤', help='character label')  # default='0123456789abcdefghijklmnopqrstuvwxyz'
        parser.add_argument('--sensitive', action='store_true',
                            help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true',
                            help='whether to keep ratio then pad for image resize')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str,  # required=True,
                            default=Transformation, help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, default=FeatureExtraction,  # required=True,
                            help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str,  # required=True,
                            default=SequenceModeling, help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str,  # required=True,
                            default=Prediction, help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20,
                            help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1,
                            help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256,
                            help='the size of the LSTM hidden state')

        # cho2:
        # self.opt = parser.parse_args()
        self.opt = parser.parse_args(args=[])

        """ vocab / character number configuration """
        if self.opt.sensitive:
            # same with ASTER setting (use 94 char).
            self.opt.character = string.printable[:-6]

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.AlignCollate_demo, self.converter = self.demo()

    def demo(self):
        """ model configuration """
        if 'CTC' in self.opt.Prediction:
            converter = CTCLabelConverter(self.opt.character)
        else:
            converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        model = Model(self.opt)
        print('model input parameters', self.opt.imgH, self.opt.imgW, self.opt.num_fiducial, self.opt.input_channel, self.opt.output_channel,
              self.opt.hidden_size, self.opt.num_class, self.opt.batch_max_length, self.opt.Transformation, self.opt.FeatureExtraction,
              self.opt.SequenceModeling, self.opt.Prediction)
        model = torch.nn.DataParallel(model).to(self.device)

        # load model
        print('loading pretrained model from %s' % self.opt.saved_model)
        model.load_state_dict(torch.load(
            self.opt.saved_model, map_location=self.device))

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        return model, AlignCollate(
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD), converter

    # predict
    @logging_time
    def predict(self, img_name_origin, image):

        # 여기서 이미지 받음
        demo_data = RawDataset(root=image,
                               opt=self.opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_demo, pin_memory=True)
        self.model.eval()

        # 여기서
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                # For max length prediction
                length_for_pred = torch.IntTensor(
                    [self.opt.batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(
                    batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

                # 여기까지가 6.6초정도
                if 'CTC' in self.opt.Prediction:
                    preds = self.model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)

                else:
                    preds = self.model(image, text_for_pred, is_train=False)
                    # for image_ in image:
                    #     for image__ in image_:
                    #         print(torch.sum(image__, axis=0) /
                    #               image__.shape[0])
                    #         print(torch.sum(image__, axis=1) /
                    #               image__.shape[1])
                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(
                        preds_index, length_for_pred)

                # log = open(f'./log_demo_result.txt', 'a')
                # dashed_line = '-' * 80
                # head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

                # print(f'{dashed_line}\n{head}\n{dashed_line}')
                # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                # pred_result = []
                # confidence_score_result = []

                # img_name,
                # for _, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                #     if 'Attn' in self.opt.Prediction:
                #         pred_EOS = pred.find('[s]')
                #         # prune after "end of sentence" token ([s])
                #         pred = pred[:pred_EOS]
                #         pred_max_prob = pred_max_prob[:pred_EOS]
                #         pred_result.append(pred)

                #     # calculate confidence score (= multiply of pred_max_prob)
                #     try:
                #         confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                #     except:
                #         confidence_score = 0
                #     confidence_score_result.append(confidence_score)
                #     print(
                #         f'{img_name_origin:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                #     log.write(
                #         f'{img_name_origin:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
                # log.close()

                preds_str = list(map(lambda x: x[:x.find('[s]')], preds_str))

                def cs(pred_max_prob):
                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    except:
                        confidence_score = 0
                    return confidence_score
                preds_max_prob = list(map(lambda x: cs(x), preds_max_prob))

                return preds_str, preds_max_prob
