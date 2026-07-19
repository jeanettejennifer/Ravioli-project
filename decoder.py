# decoder setup
from stimbposd import BPOSD
import sinter

class _BPOSDSinterCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, dem, **kwargs):
        self.decoder = BPOSD(dem, **kwargs)

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        return self.decoder.decode_batch(
            bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )

class BPOSDSinterDecoder(sinter.Decoder):
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def compile_decoder_for_dem(self, *, dem):
        return _BPOSDSinterCompiledDecoder(dem, **self.kwargs)
    
def bposd_decoder():
    custom_decoder = BPOSDSinterDecoder(
            max_bp_iters= 20,#20,
            bp_method="ms",
            schedule="serial",
            ms_scaling_factor=0.625,
            osd_method="OSD_CS",
            osd_order=5,
        )
    return custom_decoder