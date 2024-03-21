import datetime
import websockets
import ssl
import pathlib
import json
from jsonschema import validate
from jsonschema import exceptions
import jsonschema
import types

LOGTYPES = 'log', 'warning'
MESSAGE_MAXLENGTH = 100

def send_message(websocket: websockets.WebSocketClientProtocol, message: str,
                 level='log'):
    if level not in LOGTYPES:
        raise ValueError(f'Invalid log type: {level}')

    if len(message) > MESSAGE_MAXLENGTH:
        raise ValueError(f'Message is too long: {len(message)}')

    if (len(message) == 0):
        raise ValueError(f'Message is empty')

    # Check connection
    if websocket.closed:
        raise ValueError(f'Websocket connection is closed')

    message = json.dumps({
        'level': level,
        'message': message
    })

    websocket.send(message)


def finalize(websocket: websockets.WebSocketClientProtocol, msg: str):
    if (len(msg) == 0):
        raise ValueError(f'Message is empty')

    if len(msg) > MESSAGE_MAXLENGTH:
        raise ValueError(f'Message is too long: {len(msg)}')

    websocket.send(json.dumps({
        'level': 'log',
        'message': msg
    }))

    return [[], [], []]


def terminate(websocket: websockets.WebSocketClientProtocol, msg: str):
    if (len(msg) == 0):
        raise ValueError(f'Message is empty')

    if len(msg) > MESSAGE_MAXLENGTH:
        raise ValueError(f'Message is too long: {len(msg)}')

    websocket.send(json.dumps({
        'level': 'error',
        'message': msg
    }))

    return [[], [], []]


def validate_params_schema(data):
    FIELD_MINLENGTH = 1
    FIELD_MAXLENGTH = 30
    VALUE_MAXLENGTH = 100

    return True

    if (len(data) > 10):
        raise ValueError("Start params length must be max 10")

    schema = {
        "type": "array",
        "description": "Start parameters schema",
        "items": {
            "type": "object",
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "attribute": {
                            "type": "string",
                            "description": "The name of the attribute",
                            "minLength": FIELD_MINLENGTH,
                            "maxLength": FIELD_MAXLENGTH
                        },
                        "type": {
                            "const": "string",
                            "description": "The type of the attribute"
                        },
                        "defaultValue": {
                            "type": "string",
                            "description": "The default value of the attribute",
                            "minLength": FIELD_MINLENGTH,
                            "maxLength": VALUE_MAXLENGTH

                        }
                    },
                    "additionalProperties": False
                }, {
                    "type": "object",
                    "properties": {
                        "attribute": {
                            "type": "string",
                            "description": "The name of the attribute",
                            "minLength": FIELD_MINLENGTH,
                            "maxLength": FIELD_MAXLENGTH
                        },
                        "type": {
                            "const": "integer",
                            "description": "The type of the attribute"
                        },
                        "defaultValue": {
                            "type": "integer",
                            "description": "The default value of the attribute"
                        }
                    },
                    "additionalProperties": False
                }, {
                    "type": "object",
                    "properties": {
                        "attribute": {
                            "type": "string",
                            "description": "The name of the attribute",
                            "minLength": FIELD_MINLENGTH,
                            "maxLength": VALUE_MAXLENGTH
                        },
                        "type": {
                            "const": "integer",
                            "description": "The type of the attribute"
                        },
                        "defaultValue": {
                            "type": "integer",
                            "description": "The default value of the attribute"
                        },
                        "min": {
                            "type": "integer",
                            "description": "The minimum value of the attribute"
                        },
                        "max": {
                            "type": "integer",
                            "description": "The maximum value of the attribute"
                        }
                    },
                    "required": ["min", "max"],
                    "additionalProperties": False
                },
                {
                    "type": "object",
                    "properties": {
                        "attribute": {
                            "type": "string",
                            "description": "The name of the attribute",
                            "minLength": FIELD_MINLENGTH,
                            "maxLength": FIELD_MAXLENGTH
                        },
                        "type": {
                            "const": "float",
                            "description": "The type of the attribute",
                        },
                        "defaultValue": {
                            "type": "number",
                            "description": "The default value of the attribute"
                        }
                    },
                    "additionalProperties": False
                },
                {
                    "type": "object",
                    "properties": {
                        "attribute": {
                            "type": "string",
                            "description": "The name of the attribute",
                            "minLength": FIELD_MINLENGTH,
                            "maxLength": FIELD_MAXLENGTH
                        },
                        "type": {
                            "const": "float",
                            "description": "The type of the attribute",
                        },
                        "defaultValue": {
                            "type": "number",
                            "description": "The default value of the attribute"
                        },
                        "min": {
                            "type": "number",
                            "description": "The minimum value of the attribute"
                        },
                        "max": {
                            "type": "number",
                            "description": "The maximum value of the attribute"
                        },
                    },
                    "required": ["min", "max"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "attribute": {
                            "type": "string",
                            "description": "The name of the attribute",
                            "minLength": FIELD_MINLENGTH,
                            "maxLength": FIELD_MAXLENGTH
                        },
                        "type": {
                            "const": "boolean",
                            "description": "The type of the attribute"
                        },
                        "defaultValue": {
                            "type": "boolean",
                            "description": "The default value of the attribute"
                        }
                    },
                    "additionalProperties": False
                }
            ]
        }
    }

    validate(data, schema)


def process_response(connection: websockets.WebSocketClientProtocol, response,
                     simulatefun):
    try:
        jsonObj = json.loads(response)

        params = {}

        for param in jsonObj['payload']['params']:
            if (param['type'] == 'integer'):
                params[param['attribute']] = int(param['value'])
            elif (param['type'] == 'string'):
                params[param['attribute']] = str(param['value'])
            elif (param['type'] == 'float'):
                params[param['attribute']] = float(param['value'])
            elif (param['type'] == 'boolean'):
                params[param['attribute']] = bool(param['value'])

        # fun = types.FunctionType(simulatefun.__code__, {})

        res = simulatefun(connection, jsonObj['payload']['nodes'],
                  jsonObj['payload']['edges'], params, jsonObj['payload']['globals'])

        params = [res[2][param['attribute']]
                  for param in jsonObj['payload']['params']]

        return res
    except Exception as e:
        print(f'Error! {e}')


def filter_globals(fun):
    def check_global(input):
        (_, v) = input

        if (isinstance(v, types.ModuleType) or hasattr(v, '__call__')):
            return True

        # Allow classes to be imported
        if (type(v) == type):
            return True

        return False

    return dict(filter(check_global, fun.__globals__.items()))


async def connect(url: str,
                  port: int,
                  sid: str,
                  key: str,
                  title: str,
                  startParams: json,
                  simulatefun,
                  schema: json = None,
                  externalContext: ssl.SSLContext = None):
    """Connects API to the VisGraph server."""

    # Check if port is valid.
    if (port < 0 or port > 65535):
        raise ValueError("Port must be between 0 and 65535")

    #  Check if title is valid and larger than 0.
    if (len(title) > 50):
        raise ValueError("Title must be less than 50 characters")

    if (len(title) == 0):
        raise ValueError("Title must be at least 1 character")

    # Check if startParams is valid.
    validate_params_schema(startParams)

    # Filter globals
    fun = types.FunctionType(simulatefun.__code__, simulatefun.__globals__,
                             simulatefun.__name__, simulatefun.__defaults__,
                             simulatefun.__closure__)


    uri = f"wss://{url}:{int(port)}?sid={sid}&key={key}"

    print(f"Connecting to wss://{url}:{port}")

    validator = jsonschema.Validator

    try:
        validator.check_schema(schema)

        has_schema = True
    except Exception as _:
        has_schema = False

    context = True

    if (externalContext is not None):
        context = externalContext

    async with(websockets.connect(uri, max_size=2 ** 25, ssl=context)) as websocket:
        print("Connected new citadel cod")

        await websocket.send(json.dumps({
            'sessionID': sid,
            'senderType': 'simulator',
            'senderID': key,
            'type': 'registerSimulator',
            'messageSource': 'simulator',
            'receiverType': 'server',
            'receiverID': 'server',
            'payload': {
                'apikey': key,
                'params': startParams,
                'validator': has_schema,
                'title': title,
            }
        }))

        while (1):
            response = await websocket.recv()
            print(response)

            result = process_response(websocket, response, fun)

            await websocket.send(json.dumps({
                'sessionID': sid,
                'senderType': 'simulator',
                'type': 'simulatorResponse',
                'senderID': key,
                'receiverType': 'server',
                'receiverID': 'server',
                'timestamp': datetime.datetime.now().timestamp(),
                'payload': {
                    'apiKey': key,
                    'nodes': result[0],
                    'edges': result[1],
                    'params': result[2],
                    'globals': result[3],
                },
            }))


