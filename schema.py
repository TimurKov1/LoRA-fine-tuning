from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

class Listing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rooms: int | None = Field(ge=0, description='число комнат в квартире, студия = 0; null, если не указано')
    area_m2: float | None = Field(gt=0, description='площадь квартиры в квадратных метрах, числом (десятичная точка); null, если не указана')
    floor: int | None = Field(ge=1, description='этаж, на котором расположена квартира; null, если не указан')
    total_floors: int | None = Field(ge=1, description='количество этажей в доме; null, если не указан')
    metro: str | None = Field(description='название станции метро без слов «м.» и «метро»; null, если не указан')
    price_rub: int | None = Field(ge=0, description='арендная плата в рублях за месяц, целым числом; залог и коммунальные платежи не включать; null, если цена не указана или договорная')
    pets_allowed: bool | None = Field(description='true, если животные разрешены; false, если запрещены; null, если не сказано или условия неопределённые')
    phone: str | None = Field(description='телефон контактного лица, 11 цифр без пробелов и знаков, начинается с 7; null, если не указан')
    contact_name: str | None = Field(description='имя контактного лица, первая буква заглавная; null, если не указано')

    @model_validator(mode='after')
    def check_floors(self):
        if self.floor is not None and self.total_floors is not None:
            if self.floor > self.total_floors:
                raise ValueError('floor больше total_floors')
        return self

    @field_validator('phone', mode='before')
    @classmethod
    def check_phone(cls, phone):
        if phone is None:
            return None
        digits = ''.join(i for i in str(phone) if i.isdigit())

        if len(digits) == 11:
            if digits[0] == '8':
                digits = '7' + digits[1:]
            if digits[0] == '7':
                return digits
        elif len(digits) == 10:
            return '7' + digits

        raise ValueError('Номер некорректный')
