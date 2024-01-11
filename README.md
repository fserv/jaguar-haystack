# jaguar-haystack

Jaguar document store and Haystack integration


## Requirements: 

1) Jaguardb server must be running
2) Http gateway server must be running

Both servers can be started with:

```
    docker pull jaguardb/jaguardb_with_http

    docker run -d -p 8888:8888 -p 8080:8080 --name jaguardb_with_http  jaguardb/jaguardb_with_http

```

Note that docker command may require sudo to run on your system.

3) Jaguar http client package

```
    pip install -U jaguardb-http-client
```

4) Jaguar Haystack package

```
    pip install -U jaguar-haystack
```


## Documentation

    For more information, please visit:

    [www.jaguardb.com](http://www.jaguardb.com)

    [Jaguar-SDK](https://github.com/fserv/jaguar-sdk)

    [jaguar-haystack](https://pypi.org/project/jaguar-haystack/)

