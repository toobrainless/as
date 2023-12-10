import editdistance


def calc_cer(target_text, predicted_text) -> float:
    return (
        editdistance.distance(target_text, predicted_text) / len(target_text)
        if len(target_text) != 0
        else 1
    )


def calc_wer(target_text, predicted_text) -> float:
    splitted_target = target_text.split(" ")
    splitted_predicted = predicted_text.split(" ")
    return (
        editdistance.distance(splitted_target, splitted_predicted)
        / len(splitted_target)
        if len(splitted_target) != 0
        else 1
    )
